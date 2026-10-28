"""Application service boundary consumed by the local web workbench."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, cast

from photomatagent.application.history import HistoricalWorkbenchImporter
from photomatagent.application.errors import ServiceError
from photomatagent.application.models import RunRef, RunRequest, TaskCreate
from photomatagent.application.runner import RunManager
from photomatagent.application.revisions import FeedbackPlanner, RevisionService
from photomatagent.application.store import (
    ActiveRunConflict,
    HistoricalSourceConflict,
    IdempotencyConflict,
    WorkbenchStore,
)
from photomatagent.logging.session_stats import list_sessions, read_session_stats
from photomatagent.models.types import AssistantMessage, UserMessage
from photomatagent.sessions.store import (
    parse_session_snapshot,
    session_is_resumable,
    snapshot_path,
)


class WorkbenchService:
    """Composes the durable index, history importer, artifacts and run manager."""

    def __init__(
        self,
        *,
        store: WorkbenchStore,
        manager: RunManager,
        workspace_root: Path | str,
        sessions_dir: Path | str,
        artifacts: Any,
        capabilities: dict[str, dict[str, object]] | None = None,
        revision_compiler: FeedbackPlanner | None = None,
    ) -> None:
        self.store = store
        self.manager = manager
        self.workspace_root = Path(workspace_root).resolve()
        self.sessions_dir = Path(sessions_dir).resolve()
        from photomatagent.application.skills import LocalSkillService
        self.skill_service = LocalSkillService(self.workspace_root)
        manager.register_host_operation("skill_replay", self._execute_skill_replay)
        self.importer = HistoricalWorkbenchImporter(store)
        self.artifacts = artifacts
        import shutil
        import sys

        local_ready = (
            manager.local_capability_runtime_enabled
            and sys.version_info[:2] == (3, 12)
            and shutil.which("uv") is not None
        )
        self.capabilities = capabilities or {
            "local_tool_acquisition": {
                "available": local_ready,
                "reason": None if local_ready else "需要默认 Web 运行时、Python 3.12 和 uv；缺失时科研基础功能仍可使用。",
                "backend": "local_dev" if local_ready else None,
            },
            "evolution": {"available": True, "reason": None},
        }
        from photomatagent.application.qd_strategy import QDStrategyService
        self.qd_service = QDStrategyService(store=store, manager=manager, workspace=self.workspace_root)
        from photomatagent.application.evaluations import EvaluationService
        self.evaluation_service = EvaluationService(
            store=store, manager=manager, workspace_root=self.workspace_root,
        )
        self.revision_service = RevisionService(
            store=store,
            manager=manager,
            artifacts=artifacts,
            compiler=revision_compiler,
            qd_service=self.qd_service,
        )

    def skill_sources(self) -> list[dict[str, Any]]:
        import json
        from photomatagent.scientific.evolution.qd.store import QDStore, QDStoreError
        from photomatagent.capability_acquisition.models import ExecutionRecord, ExecutionRequest
        from photomatagent.errors import ToolExecutionError
        items = []
        for source in self.skill_service.sources():
            try:
                run = self.store.run(str(source["run_id"]))
                if run.task_id == source["task_id"] and run.status in {"SUCCEEDED", "PARTIAL"}:
                    items.append(source | {"kind": "ordinary_run"})
            except KeyError:
                continue
        for task in self.store.tasks():
            for link in self.store.links_for_task(task.task_id):
                if link.source_type != "qd_trial":
                    continue
                try:
                    qd_store = QDStore(self.skill_service.workspace, trial_factory=lambda event, genome: (_ for _ in ()).throw(ValueError("READ_ONLY")))
                    trial = qd_store.load_trial(link.source_id)
                    if trial.status not in {"COLLECTED", "VALIDATED"}:
                        continue
                    root = self.skill_service.workspace.resolve(trial.attempt_dir + "/.photomatagent/executions", must_exist=False)
                    for directory in root.iterdir() if root.exists() else []:
                        if directory.is_symlink() or not directory.is_dir():
                            continue
                        record = ExecutionRecord.model_validate_json((directory / "record.json").read_text())
                        request = ExecutionRequest.model_validate(json.loads((directory / "request.json").read_text())["request"])
                        if request.trial_id == trial.trial_id:
                            items.append({"kind": "qd_trial", "execution_id": record.execution_id, "task_id": task.task_id,
                                          "trial_id": trial.trial_id, "script_path": request.script_path,
                                          "input_keys": list(request.input_refs), "artifacts": [a.path for a in record.artifacts], "state": record.state})
                except (QDStoreError, ValueError, OSError, ToolExecutionError):
                    continue
        return items

    async def start_skill_replay(self, skill_id: str, payload: Any, request_key: str) -> RunRef:
        from photomatagent.application.skills import SkillReplay
        self.task_detail(payload.task_id)
        self.skill_service.read(skill_id)
        refs = payload.input_refs
        if refs is not None:
            resolved = {}
            for key, ref in refs.items():
                if ref.startswith("artifact:"):
                    try:
                        row = self.store.artifact(ref.removeprefix("artifact:"))
                    except KeyError as exc:
                        raise ServiceError("SKILL_INPUT_UNAVAILABLE", "输入文件不存在。") from exc
                    if row["task_id"] != payload.task_id:
                        raise ServiceError("SKILL_INPUT_SCOPE_VIOLATION", "输入不属于当前任务。")
                    resolved[key] = row["path"]
                else:
                    resolved[key] = ref
            payload = SkillReplay(task_id=payload.task_id, scope=payload.scope, input_refs=resolved, argv=payload.argv)
        request = RunRequest(text="回放冻结技能方法", strategy="skill_replay",
                             host_operation={"operation": "skill_replay", "skill_id": skill_id, "payload": payload.model_dump()})
        try:
            reference = self.store.reserve_run(payload.task_id, request, request_key)
            if self.store.run(reference.run_id).status == "QUEUED":
                await self.manager.enqueue(reference.run_id)
            return reference
        except ActiveRunConflict as exc:
            raise ServiceError("ACTIVE_RUN_EXISTS", str(exc), status_code=409) from exc
        except IdempotencyConflict as exc:
            raise ServiceError("IDEMPOTENCY_CONFLICT", str(exc), status_code=409) from exc

    async def _execute_skill_replay(self, run: Any) -> None:
        from time import monotonic
        from uuid import uuid4
        import mimetypes
        from photomatagent.application.skills import SkillReplay
        from photomatagent.capability_acquisition.sandbox import _sha256_file
        operation = run.request.host_operation
        if operation is None or operation["operation"] != "skill_replay" or run.request.strategy != "skill_replay":
            raise ValueError("SKILL_HOST_OPERATION_INVALID")
        payload = SkillReplay.model_validate(operation["payload"])
        if payload.task_id != run.task_id:
            raise ValueError("SKILL_HOST_TASK_MISMATCH")
        started = monotonic()
        result = await self.skill_service.replay(str(operation["skill_id"]), payload, run_id=run.run_id)
        for relative in result["artifacts"]:
            path = self.skill_service.workspace.resolve(relative)
            self.store.register_artifact(artifact_id=f"artifact_{uuid4().hex}", task_id=payload.task_id,
                                         run_id=run.run_id, path=relative, filename=path.name, sha256=_sha256_file(path),
                                         media_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream", size_bytes=path.stat().st_size)
        self.store.append_event_for_run(run.run_id, "skill_use", result | {"wall_seconds": monotonic() - started, "model_calls": 0})
        self.store.append_event_for_run(run.run_id, "loop_completed", {"reason": "completed" if result["code"] == "OK" and result["checker"]["status"] == "PASS" else "unresolved"})

    def create_task(self, request: TaskCreate) -> str:
        if request.attachment_ids:
            raise ServiceError("INVALID_ATTACHMENT", "新建任务不能引用其他任务的附件。")
        return self.store.create_task(request)

    def list_tasks(self) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for task in self.store.tasks():
            runs = self.store.runs_for_task(task.task_id)
            result.append(task.model_dump(mode="json") | {
                "latest_run": runs[-1].model_dump(mode="json") if runs else None,
            })
        return result

    def task_detail(self, task_id: str) -> dict[str, object]:
        try:
            task = self.store.task(task_id)
            runs = self.store.runs_for_task(task_id)
            links = self.store.links_for_task(task_id)
            artifacts = self.store.artifacts_for_task(task_id)
        except KeyError as error:
            raise ServiceError("NOT_FOUND", "任务不存在。", status_code=404) from error
        journal = self.store.events_after(task_id, 0)
        progress = [event for event in journal if event.type == "closed_loop_completed"]
        candidates = [event.payload for event in journal if event.type == "evolution_candidate_captured"]
        frozen_versions = {
            str(item["bundle_sha256"])
            for record in self.store.evaluations_for_task(task_id)
            for item in cast(dict[str, Any], record.get("manifest") or {}).get("frozen_skills", [])
            if isinstance(item, dict) and item.get("bundle_sha256")
        }
        return {
            "task": task.model_dump(mode="json"),
            "runs": [run.model_dump(mode="json") for run in runs],
            "links": [link.model_dump(mode="json") for link in links],
            "artifacts": [self.artifacts.metadata_row(row) for row in artifacts],
            "evolution": {
                "latest": progress[-1].payload if progress else None,
                "candidates": candidates,
                "skill_reliability": [self.evaluation_service.skill_reliability(digest, task_id=task_id) for digest in sorted(frozen_versions)],
            },
        }

    def historical_task(self, task_id: str) -> dict[str, object]:
        try:
            self.store.task(task_id)
            link = next(
                (
                    item
                    for item in self.store.links_for_task(task_id)
                    if item.source_type == "runtime_session"
                ),
                None,
            )
        except KeyError as error:
            raise ServiceError("NOT_FOUND", "任务不存在。", status_code=404) from error
        if link is None:
            raise ServiceError("NOT_FOUND", "此任务没有关联的历史会话。", status_code=404)
        if Path(link.source_id).name != link.source_id or link.source_id in {".", ".."}:
            raise ServiceError("INVALID_SESSION_ID", "历史会话来源无效。", status_code=409)
        source = (self.sessions_dir / link.source_id).resolve()
        try:
            source.relative_to(self.sessions_dir)
        except ValueError as error:
            raise ServiceError("INVALID_SESSION_ID", "历史会话来源路径越界。", status_code=409) from error
        events_path = source / "events.jsonl"
        try:
            current_digest = hashlib.sha256(events_path.read_bytes()).hexdigest()
        except OSError as error:
            raise ServiceError("HISTORY_SOURCE_MISSING", "原始历史会话已不可读取。", status_code=409) from error
        messages: list[dict[str, str]] = []
        truncated = False
        state_path = snapshot_path(source)
        snapshot_payload: bytes | None = None
        try:
            snapshot_payload = state_path.read_bytes() if state_path.is_file() else None
            snapshot_digest = hashlib.sha256(snapshot_payload).hexdigest() if snapshot_payload is not None else None
        except OSError:
            snapshot_digest = None
        messages_current = (
            None
            if link.snapshot_digest is None
            else snapshot_digest == link.snapshot_digest
        )
        if messages_current:
            try:
                assert snapshot_payload is not None
                snapshot = parse_session_snapshot(snapshot_payload)
                for item in snapshot.conversation.messages:
                    if isinstance(item, UserMessage):
                        role, text = "user", item.content
                    elif isinstance(item, AssistantMessage):
                        role, text = "assistant", item.text
                    else:
                        continue
                    if len(messages) >= 200:
                        truncated = True
                        break
                    messages.append({"role": role, "text": text[:12_000]})
                    truncated = truncated or len(text) > 12_000
            except (OSError, ValueError):
                # A malformed snapshot is not shown as trusted imported history.
                messages_current = False
                messages = []
        return {
            "source": {
                "session_id": link.source_id,
                "source_digest": link.source_digest,
                "current": current_digest == link.source_digest,
                "messages_current": messages_current,
            },
            "messages": messages,
            "messages_truncated": truncated,
        }

    async def start_run(self, task_id: str, request: RunRequest, request_key: str) -> RunRef:
        self._require_available_strategy(request)
        try:
            reference = self.store.reserve_run(task_id, request, request_key)
            if self.store.run(reference.run_id).status == "QUEUED":
                await self.manager.enqueue(reference.run_id)
            return reference
        except KeyError as error:
            raise ServiceError("NOT_FOUND", "任务不存在。", status_code=404) from error
        except IdempotencyConflict as error:
            raise ServiceError("IDEMPOTENCY_CONFLICT", str(error), status_code=409) from error
        except ActiveRunConflict as error:
            raise ServiceError("ACTIVE_RUN_EXISTS", str(error), status_code=409, retryable=True) from error

    async def cancel_run(self, run_id: str) -> str:
        try:
            return await self.manager.cancel(run_id)
        except KeyError as error:
            raise ServiceError("NOT_FOUND", "运行不存在。", status_code=404, run_id=run_id) from error

    async def resume_run(self, run_id: str, request_key: str) -> RunRef:
        try:
            previous = self.store.run(run_id)
            if previous.status != "INTERRUPTED":
                raise ServiceError(
                    "RUN_NOT_INTERRUPTED",
                    "只有中断的运行可以继续。",
                    status_code=409,
                    run_id=run_id,
                    next_action="创建新的运行，或先处理当前运行状态。",
                )
            host_operation = previous.request.host_operation
            if host_operation is None:
                self._require_available_strategy(previous.request, run_id=run_id)
            else:
                operation_name = host_operation.get("operation")
                supported = (
                    previous.request.strategy in {"qd", "skill_replay"}
                    and isinstance(operation_name, str)
                    and self.manager.has_host_operation(operation_name)
                )
                if not supported:
                    raise ServiceError(
                        "STRATEGY_UNAVAILABLE",
                        f"中断的 {previous.request.strategy} 运行没有可用的可信恢复处理器。",
                        status_code=409,
                        run_id=run_id,
                        next_action="检查后端能力状态；不可恢复时保留原运行并创建普通反馈修订。",
                    )
            recovery = await self.manager.recover(run_id)
            if recovery.status.value == "RECONCILIATION_REQUIRED":
                raise ServiceError(
                    "RECONCILIATION_REQUIRED",
                    recovery.message,
                    status_code=409,
                    run_id=run_id,
                    next_action="先核实外部作业状态与产物，再创建后续运行。",
                )
            reference = self.store.reserve_run(
                previous.task_id,
                previous.request,
                request_key,
                operation="resume_run",
                idempotency_binding={"source_run_id": run_id},
            )
            if reference.run_id == run_id:
                raise ServiceError(
                    "IDEMPOTENCY_CONFLICT",
                    "此请求键已用于创建原运行，继续运行需要一个新的请求键。",
                    status_code=409,
                    run_id=run_id,
                    next_action="为继续运行生成新的 Idempotency-Key。",
                )
            if self.store.run(reference.run_id).status == "QUEUED":
                await self.manager.enqueue(reference.run_id)
            return reference
        except KeyError as error:
            raise ServiceError("NOT_FOUND", "运行不存在。", status_code=404, run_id=run_id) from error
        except IdempotencyConflict as error:
            raise ServiceError("IDEMPOTENCY_CONFLICT", str(error), status_code=409, run_id=run_id) from error
        except ActiveRunConflict as error:
            raise ServiceError("ACTIVE_RUN_EXISTS", str(error), status_code=409, run_id=run_id, retryable=True) from error

    @staticmethod
    def _require_available_strategy(request: RunRequest, *, run_id: str | None = None) -> None:
        if request.strategy != "standard" or request.host_operation is not None:
            raise ServiceError(
                "STRATEGY_UNAVAILABLE",
                f"当前阶段尚未接入 {request.strategy} 策略，未启动运行。",
                status_code=409,
                run_id=run_id,
                next_action="切换为普通科研运行；反馈与 QD 策略将在对应阶段开放。",
            )

    def events_after(self, task_id: str, seq: int):
        try:
            self.store.task(task_id)
            return self.store.events_after(task_id, seq, limit=500)
        except KeyError as error:
            raise ServiceError("NOT_FOUND", "任务不存在。", status_code=404) from error

    def history_sessions(self) -> list[dict[str, object]]:
        sessions: list[dict[str, object]] = []
        for path in list_sessions(self.sessions_dir):
            try:
                stats = read_session_stats(path)
            except (OSError, ValueError):
                continue
            sessions.append(
                {
                    "session_id": path.name,
                    "resumable": session_is_resumable(path),
                    "provider": stats.provider,
                    "model": stats.model,
                    "iterations": stats.iterations,
                    "tool_calls": stats.tool_calls,
                    "stop_reason": stats.stop_reason,
                }
            )
        return sessions

    def import_history(self, session_id: str) -> str:
        if not session_id or Path(session_id).name != session_id or session_id in {".", ".."}:
            raise ServiceError("INVALID_SESSION_ID", "历史会话 ID 格式无效。")
        source = (self.sessions_dir / session_id).resolve()
        try:
            source.relative_to(self.sessions_dir)
        except ValueError as error:
            raise ServiceError("INVALID_SESSION_ID", "历史会话路径越界。") from error
        if not source.is_dir():
            raise ServiceError("NOT_FOUND", "历史会话不存在。", status_code=404)
        try:
            return self.importer.import_session(source)
        except HistoricalSourceConflict as error:
            raise ServiceError(
                "HISTORICAL_SOURCE_CONFLICT",
                "此历史会话自上次导入后已发生变化，未覆盖已导入记录。",
                status_code=409,
                next_action="请以新来源副本创建任务，或检查原始会话后重试。",
            ) from error
        except (OSError, ValueError) as error:
            raise ServiceError("HISTORY_IMPORT_FAILED", str(error), status_code=422) from error

    def capability_list(self) -> list[dict[str, object]]:
        return [
            {"id": key, **value}
            for key, value in sorted(self.capabilities.items())
        ]


__all__ = ["ServiceError", "WorkbenchService"]
