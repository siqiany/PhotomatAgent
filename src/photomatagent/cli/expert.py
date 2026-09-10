"""Interactive, in-chat expert review wizard."""

from __future__ import annotations

from pathlib import Path
from typing import Awaitable, Callable, Protocol

from rich.console import Console

from photomatagent.cli.evolve import run_compile_command, run_feedback_command
from photomatagent.observability.trace import (
    TraceError,
    list_session_paths,
    resolve_session_path,
)
from photomatagent.runtime.loop import AgentRuntime
from photomatagent.scientific.evolution.importer import (
    HistoricalSessionImporter,
    preview_historical_session,
)
from photomatagent.scientific.evolution.service import EvolutionService
from photomatagent.scientific.evolution.store import EvolutionStore
from photomatagent.scientific.loop import TargetSpec
from photomatagent.sessions.store import save_session_snapshot
from photomatagent.workspace import Workspace


class PromptSessionLike(Protocol):
    async def prompt_async(self, message: str) -> str: ...


class _ExpertPrompt:
    """Prefix every wizard prompt while preserving the feedback protocol."""

    def __init__(self, session: PromptSessionLike) -> None:
        self._session = session

    async def prompt_async(self, message: str) -> str:
        if message.startswith("[EXPERT MODE |"):
            prompt = message
        else:
            prompt = f"[EXPERT MODE | FEEDBACK] {message}"
        return await self._session.prompt_async(prompt)


async def _ask(session: PromptSessionLike, label: str) -> str:
    return await session.prompt_async(f"[EXPERT MODE | {label}] ")


async def run_expert_mode(
    *,
    session: PromptSessionLike,
    output: Console,
    workspace: Path | str,
    source: str | Path | None = None,
    sessions_dir: Path | str | None = None,
    logger: object | None = None,
    runtime: AgentRuntime | None = None,
    compile_runner: Callable[..., Awaitable[object]] = run_compile_command,
    iterate_callback: Callable[[str], Awaitable[object]] | None = None,
) -> None:
    """Run the bounded expert wizard; all input is consumed by this function."""
    boundary = Workspace(workspace)
    if source is None or str(source).lower() == "current":
        if logger is not None and runtime is not None:
            # Preserve the live session before inspecting/importing it.
            save_session_snapshot(
                logger.session_dir,  # type: ignore[attr-defined]
                conversation=runtime.conversation_state,
                scientific=runtime.scientific_state,
                engine=runtime.context_engine.snapshot(),
            )
            source_path = logger.session_dir  # type: ignore[attr-defined]
        elif runtime is not None and runtime.session_id:
            source_path = boundary.resolve(
                f".photomatagent/sessions/{runtime.session_id}", must_exist=True
            )
        else:
            output.print("[EXPERT MODE | ERROR] 当前 session 没有可检查的持久化记录。")
            return
    elif str(source).lower() == "history":
        paths = list_session_paths(sessions_dir)
        if not paths:
            output.print("[EXPERT MODE | ERROR] 没有可用的历史 session。")
            return
        output.print("[EXPERT MODE | HISTORY] 最近 session：" + ", ".join(p.name for p in paths[:20]))
        selected = (await _ask(session, "选择 session-id，或 /cancel" )).strip()
        if selected.lower() == "/cancel":
            output.print("[EXPERT MODE | CANCELLED] 已取消。")
            return
        try:
            source_path = resolve_session_path(selected, sessions_dir)
        except TraceError as exc:
            output.print(f"[EXPERT MODE | ERROR] {exc}")
            return
    else:
        try:
            source_path = resolve_session_path(str(source), sessions_dir)
        except TraceError as exc:
            output.print(f"[EXPERT MODE | ERROR] {exc}")
            return

    try:
        preview = preview_historical_session(boundary, source_path)
        output.print(f"[EXPERT MODE | GOAL] {preview.goal}")
        goal_confirmation = (await _ask(session, "确认 goal（y/n）或 /cancel")).strip().lower()
        if goal_confirmation == "/cancel":
            output.print("[EXPERT MODE | CANCELLED] 已取消。")
            return
        goal = preview.goal
        if goal_confirmation not in {"y", "yes"}:
            replacement = await _ask(session, "输入替换 goal，或 /cancel")
            if replacement.strip().lower() == "/cancel":
                output.print("[EXPERT MODE | CANCELLED] 已取消。")
                return
            if not replacement.strip():
                raise ValueError("goal replacement cannot be empty")
            goal = replacement.strip()
        output.print(f"[EXPERT MODE | RESULT] {preview.final_response}")
        result_confirmation = (await _ask(session, "确认 result（y/n），或 /cancel")).strip().lower()
        if result_confirmation.strip().lower() == "/cancel":
            output.print("[EXPERT MODE | CANCELLED] 已取消。")
            return
        artifact_path: Path | None = None
        if result_confirmation not in {"y", "yes"}:
            replacement_result = await _ask(session, "输入替换 result，或 /cancel")
            if replacement_result.strip().lower() == "/cancel":
                output.print("[EXPERT MODE | CANCELLED] 已取消。")
                return
            if not replacement_result.strip():
                raise ValueError("result replacement cannot be empty")
            artifact_path = boundary.resolve(replacement_result.strip(), must_exist=True)
            if not artifact_path.is_file():
                raise ValueError("replacement artifact must be a regular file")
        target_name = (await _ask(session, "workspace-contained TargetSpec JSON 文件路径，或 /cancel")).strip()
        if target_name.lower() == "/cancel":
            output.print("[EXPERT MODE | CANCELLED] 已取消。")
            return
        target_path = boundary.resolve(target_name, must_exist=True)
        if not target_path.is_file():
            raise ValueError("TargetSpec path must be a regular file")
        target = TargetSpec.model_validate_json(target_path.read_text(encoding="utf-8"))
        if not target.constraints:
            raise ValueError("TargetSpec must contain at least one constraint")
        output.print(f"[EXPERT MODE | TARGET] {target.model_dump_json()}")
        target_confirmation = (await _ask(session, "确认 target（y/n）或 /cancel")).strip().lower()
        if target_confirmation == "/cancel":
            output.print("[EXPERT MODE | CANCELLED] 已取消。")
            return
        if target_confirmation not in {"y", "yes"}:
            output.print("[EXPERT MODE | CANCELLED] 未确认 target，未写入数据。")
            return
        target = target.model_copy(update={"goal": goal})
        service = EvolutionService(EvolutionStore(boundary))
        task = HistoricalSessionImporter(boundary, service).import_session(
            source_path, target=target, goal=goal, artifact_path=artifact_path
        )
        selected_version = task.last_completed_version or task.current_version or "v001"
        output.print(f"[EXPERT MODE | IMPORTED] {task.evolution_id} {selected_version}")
        if task.status != "AWAITING_EXPERT_FEEDBACK":
            if task.status == "FEEDBACK_RECORDED":
                choice = (await _ask(session, "已有反馈；立即 compile？[y/N] 或 /cancel")).strip().lower()
                if choice == "/cancel" or choice not in {"y", "yes"}:
                    return
                await compile_runner(
                    session=_ExpertPrompt(session), output=output, workspace=boundary.root,
                    evolution_id=task.evolution_id,
                )
                return
            if task.status == "REVISION_READY":
                choice = (await _ask(session, "revision confirmed；立即 iterate？[y/N] 或 /cancel")).strip().lower()
                if choice == "/cancel" or choice not in {"y", "yes"}:
                    return
                if iterate_callback is not None:
                    await iterate_callback(task.evolution_id)
                return
            output.print(f"[EXPERT MODE | ERROR] task status {task.status} cannot resume")
            return
        record = await run_feedback_command(
            session=_ExpertPrompt(session), output=output, workspace=boundary.root,
            evolution_id=task.evolution_id, version=selected_version,
        )
        if record is None:
            return
        compile_choice = (await _ask(session, "立即 compile？[y/N] 或 /cancel")).strip().lower()
        if compile_choice == "/cancel":
            output.print("[EXPERT MODE | CANCELLED] 已取消后续操作；反馈已保留。")
            return
        if compile_choice not in {"y", "yes"}:
            return
        await compile_runner(
            session=_ExpertPrompt(session), output=output, workspace=boundary.root,
            evolution_id=task.evolution_id, version=selected_version,
        )
        refreshed = service.get(task.evolution_id)
        if refreshed.status == "REVISION_READY":
            iterate_choice = (await _ask(session, "revision confirmed；立即 iterate？[y/N] 或 /cancel")).strip().lower()
            if iterate_choice == "/cancel":
                output.print("[EXPERT MODE | CANCELLED] 已取消 iterate。")
            elif iterate_choice in {"y", "yes"} and iterate_callback is not None:
                await iterate_callback(task.evolution_id)
    except (OSError, UnicodeError, ValueError, TraceError) as exc:
        output.print(f"[EXPERT MODE | ERROR] {exc}")


__all__ = ["run_expert_mode"]
