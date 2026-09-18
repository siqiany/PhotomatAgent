"""Interactive, in-chat expert review wizard."""

from __future__ import annotations

from pathlib import Path
from typing import Awaitable, Callable, Protocol

from rich.console import Console
from rich.table import Table

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
from photomatagent.scientific.evolution.models import EvolutionTask
from photomatagent.scientific.evolution.service import EvolutionService
from photomatagent.scientific.evolution.store import EvolutionStore
from photomatagent.scientific.evolution.targeting import (
    ConfirmedTargetStore,
    TargetSpecCompiler,
    TargetSpecDraft,
    TargetTaskKind,
)
from photomatagent.scientific.discovery import DiscoveryConstraints
from photomatagent.scientific.loop import TargetSpec
from photomatagent.scientific.state import ScientificState
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


def _bind_authoritative_target(
    target: TargetSpec,
    *,
    goal: str,
    task_kind: TargetTaskKind,
    discovery_constraints: DiscoveryConstraints,
) -> TargetSpec:
    """Bind caller-owned target fields after every non-compiler target load."""

    metadata = dict(target.metadata)
    metadata["task_kind"] = task_kind
    metadata["discovery"] = discovery_constraints.model_dump(mode="json")
    return target.model_copy(update={"goal": goal, "metadata": metadata})


async def _ask(session: PromptSessionLike, label: str) -> str:
    return await session.prompt_async(f"[EXPERT MODE | {label}] ")


def _render_target(output: Console, target: TargetSpec, draft: TargetSpecDraft | None) -> None:
    table = Table("#", "Property", "Rule", "Severity", "Basis / rationale")
    for index, constraint in enumerate(target.constraints, start=1):
        detail = "已确认缓存或文件导入"
        if draft is not None and index <= len(draft.constraints):
            proposed = draft.constraints[index - 1]
            detail = (
                f"{proposed.basis}; confidence={proposed.confidence:.2f}; "
                f"{proposed.rationale}"
            )
        table.add_row(
            str(index),
            constraint.property,
            f"{constraint.operator} {constraint.value} {constraint.unit}".strip(),
            constraint.severity,
            detail,
        )
    output.print(table)
    if draft is not None:
        for warning in draft.warnings:
            output.print(f"[EXPERT MODE | TARGET WARNING] {warning}")


async def _automatic_target(
    *,
    session: PromptSessionLike,
    output: Console,
    boundary: Workspace,
    session_id: str,
    goal: str,
    scientific_state: ScientificState,
    compiler: TargetSpecCompiler,
    task_kind: TargetTaskKind = "validation",
    discovery_constraints: DiscoveryConstraints | None = None,
) -> TargetSpec | None:
    """Load or generate a target, require confirmation, and cache it."""

    store = ConfirmedTargetStore(boundary)
    discovery = DiscoveryConstraints.model_validate(discovery_constraints or {})
    cached = store.load(
        session_id,
        goal=goal,
        scientific_state=scientific_state,
        task_kind=task_kind,
        discovery_constraints=discovery,
    )
    draft: TargetSpecDraft | None = None
    if cached is not None:
        target = _bind_authoritative_target(
            cached.target,
            goal=goal,
            task_kind=task_kind,
            discovery_constraints=discovery,
        )
        draft = cached.draft
        output.print("[EXPERT MODE | TARGET] 已自动加载该历史任务确认过的 TargetSpec。")
    else:
        output.print("[EXPERT MODE | TARGET] 未找到已确认目标，正在隔离、无工具地自动生成草案……")
        target = None

    correction: str | None = None
    while True:
        if target is None:
            try:
                draft = await compiler.compile(
                    goal=goal,
                    scientific_state=scientific_state,
                    correction=correction,
                    task_kind=task_kind,
                    discovery_constraints=discovery_constraints,
                )
                target = draft.target
            except ValueError as exc:
                output.print(f"[EXPERT MODE | ERROR] {exc}")
                fallback = (
                    await _ask(session, "自动生成失败；输入 TargetSpec JSON 文件路径，或 /cancel")
                ).strip()
                if fallback.lower() == "/cancel":
                    output.print("[EXPERT MODE | CANCELLED] 已取消。")
                    return None
                target_path = boundary.resolve(fallback, must_exist=True)
                if not target_path.is_file():
                    raise ValueError("TargetSpec path must be a regular file")
                target = _bind_authoritative_target(
                    TargetSpec.model_validate_json(target_path.read_text(encoding="utf-8")),
                    goal=goal,
                    task_kind=task_kind,
                    discovery_constraints=discovery,
                )
                draft = None
        target = _bind_authoritative_target(
            target,
            goal=goal,
            task_kind=task_kind,
            discovery_constraints=discovery,
        )
        if task_kind == "validation" and not target.constraints:
            raise ValueError("TargetSpec must contain at least one constraint")
        _render_target(output, target, draft)
        choice = (
            await _ask(
                session,
                "确认 target：[y]确认 [e]告诉智能体如何修改 [r]重新生成 [f]文件导入 [c]取消",
            )
        ).strip().lower()
        if choice in {"y", "yes"}:
            model_provider = compiler.model
            store.save(
                session_id=session_id,
                goal=goal,
                scientific_state=scientific_state,
                target=target,
                draft=draft,
                provider=str(getattr(model_provider, "provider", "unknown")),
                model=str(getattr(model_provider, "model", "unknown")),
                task_kind=task_kind,
                discovery_constraints=discovery,
            )
            output.print("[EXPERT MODE | TARGET] 已确认并自动保存；下次评价该任务会直接加载。")
            return target
        if choice in {"c", "/cancel"}:
            output.print("[EXPERT MODE | CANCELLED] 未确认 target，未写入数据。")
            return None
        if choice == "e":
            correction = (await _ask(session, "描述需要增加、删除或修改的评价标准，或 /cancel")).strip()
            if correction.lower() == "/cancel":
                return None
            if not correction:
                output.print("[EXPERT MODE | TARGET] 修改要求不能为空。")
                continue
            target = None
            continue
        if choice == "r":
            correction = "Regenerate an alternative draft while obeying the same goal and safety rules."
            target = None
            continue
        if choice == "f":
            target_name = (await _ask(session, "workspace-contained TargetSpec JSON 文件路径，或 /cancel")).strip()
            if target_name.lower() == "/cancel":
                return None
            target_path = boundary.resolve(target_name, must_exist=True)
            if not target_path.is_file():
                raise ValueError("TargetSpec path must be a regular file")
            target = _bind_authoritative_target(
                TargetSpec.model_validate_json(target_path.read_text(encoding="utf-8")),
                goal=goal,
                task_kind=task_kind,
                discovery_constraints=discovery,
            )
            draft = None
            correction = None
            continue
        output.print("[EXPERT MODE | TARGET] 请输入 y、e、r、f 或 c。")


async def _resume_linked(
    *, task: EvolutionTask, session: PromptSessionLike, output: Console,
    workspace: Path, compile_runner: Callable[..., Awaitable[object]],
    iterate_callback: Callable[[str], Awaitable[object]] | None,
    service: EvolutionService,
) -> None:
    status = task.status
    evolution_id = task.evolution_id
    version = task.last_completed_version or task.current_version
    if status == "AWAITING_EXPERT_FEEDBACK" and version is not None:
        record = await run_feedback_command(
            session=_ExpertPrompt(session), output=output, workspace=workspace,
            evolution_id=evolution_id, version=version,
        )
        if record is None:
            return
        await _compile_then_maybe_iterate(
            task=task, session=session, output=output, workspace=workspace,
            compile_runner=compile_runner, iterate_callback=iterate_callback,
            service=service, ask_compile=True,
        )
    elif status == "FEEDBACK_RECORDED":
        await _compile_then_maybe_iterate(
            task=task, session=session, output=output, workspace=workspace,
            compile_runner=compile_runner, iterate_callback=iterate_callback,
            service=service, ask_compile=True,
        )
    elif status == "REVISION_READY":
        choice = (await _ask(session, "revision confirmed；立即 iterate？[y/N] 或 /cancel")).strip().lower()
        if choice in {"y", "yes"} and iterate_callback is not None:
            await iterate_callback(evolution_id)


async def _compile_then_maybe_iterate(
    *, task: EvolutionTask, session: PromptSessionLike, output: Console,
    workspace: Path, compile_runner: Callable[..., Awaitable[object]],
    iterate_callback: Callable[[str], Awaitable[object]] | None,
    service: EvolutionService, ask_compile: bool,
) -> None:
    if ask_compile:
        choice = (await _ask(session, "已有反馈；立即 compile？[y/N] 或 /cancel")).strip().lower()
        if choice not in {"y", "yes"}:
            return
    await compile_runner(session=_ExpertPrompt(session), output=output,
                         workspace=workspace, evolution_id=task.evolution_id)
    refreshed = service.get(task.evolution_id)
    if refreshed.status == "REVISION_READY":
        choice = (await _ask(session, "revision confirmed；立即 iterate？[y/N] 或 /cancel")).strip().lower()
        if choice in {"y", "yes"} and iterate_callback is not None:
            await iterate_callback(task.evolution_id)


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
    target_compiler: TargetSpecCompiler | None = None,
    task_kind: TargetTaskKind = "validation",
    discovery_constraints: DiscoveryConstraints | None = None,
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
        service = EvolutionService(EvolutionStore(boundary))
        importer = HistoricalSessionImporter(boundary, service)
        linked = importer.find_linked_task(preview.session_id)
        if linked is not None:
            output.print(f"[EXPERT MODE | RESUMING] {linked.evolution_id} {linked.status}")
            await _resume_linked(task=linked, session=session, output=output,
                                 workspace=boundary.root, compile_runner=compile_runner,
                                 iterate_callback=iterate_callback, service=service)
            return
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
        scientific_state = (
            preview.snapshot.scientific
            if preview.snapshot is not None
            else ScientificState(goal=goal)
        )
        if target_compiler is None:
            if runtime is None:
                raise ValueError("automatic TargetSpec generation requires the current model provider")
            target_compiler = TargetSpecCompiler(runtime.model_provider)
        target = await _automatic_target(
            session=session,
            output=output,
            boundary=boundary,
            session_id=preview.session_id,
            goal=goal,
            scientific_state=scientific_state,
            compiler=target_compiler,
            task_kind=task_kind,
            discovery_constraints=discovery_constraints,
        )
        if target is None:
            return
        task = importer.import_session(
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
