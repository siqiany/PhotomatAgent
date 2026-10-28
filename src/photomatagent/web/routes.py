"""HTTP routes for task execution, event replay, history and artifacts."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
import tempfile
from time import monotonic

from fastapi import APIRouter, File, Header, Request, UploadFile
from starlette.background import BackgroundTask
from fastapi.responses import StreamingResponse

from photomatagent.application.skills import SkillCreate, SkillReplay
from photomatagent.application.procedural_skills import ProcedureCreate
from photomatagent.application.procedure_workflow import ProcedureReview, ProcedureWorkflow
from photomatagent.application.models import FeedbackRequest, RunRequest, TaskCreate
from photomatagent.application.errors import ServiceError
from photomatagent.application.service import WorkbenchService
from photomatagent.web.schemas import (
    ApprovalDecisionRequest,
    ApprovalDecisionResponse,
    ArtifactMetadataResponse,
    AttachmentResponse,
    CancelRunResponse,
    CapabilityListResponse,
    EvaluationCapabilitiesResponse,
    EvaluationStartRequest,
    EvaluationStartResponse,
    EvaluationStatusResponse,
    HistoryImportRequest,
    HistorySessionListResponse,
    HistoricalTaskResponse,
    RunRefResponse,
    RevisionPlanResponse,
    RevisionRunRequest,
    TaskDetailResponse,
    TaskIdResponse,
    TaskListResponse,
)

router = APIRouter(prefix="/api")
_SSE_POLL_SECONDS = 0.25
_SSE_HEARTBEAT_SECONDS = 15.0


def _request_key(value: str | None) -> str:
    if value is None or not value.strip():
        raise ServiceError(
            "IDEMPOTENCY_KEY_REQUIRED",
            "请求必须包含 Idempotency-Key。",
            status_code=400,
            next_action="为本次操作生成唯一请求键后重试。",
        )
    if len(value) > 200:
        raise ServiceError("INVALID_IDEMPOTENCY_KEY", "Idempotency-Key 不能超过 200 个字符。")
    return value


def _service(request: Request) -> WorkbenchService:
    return request.app.state.service


@router.get("/evaluations/capabilities", response_model=EvaluationCapabilitiesResponse)
def evaluation_capabilities(request: Request) -> dict[str, object]:
    return _service(request).evaluation_service.capabilities()


@router.post("/evaluations", status_code=202, response_model=EvaluationStartResponse)
async def start_evaluation(
    payload: EvaluationStartRequest,
    request: Request,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> dict[str, object]:
    response = await _service(request).evaluation_service.start(payload, _request_key(idempotency_key))
    return response.model_dump(mode="json")


@router.get("/tasks/{task_id}/evaluations", response_model=list[EvaluationStatusResponse])
def task_evaluations(task_id: str, request: Request) -> list[dict[str, object]]:
    return [item.model_dump(mode="json") for item in _service(request).evaluation_service.list_for_task(task_id)]


@router.get("/evaluations/{evaluation_id}", response_model=EvaluationStatusResponse)
def evaluation_status(evaluation_id: str, request: Request) -> dict[str, object]:
    return _service(request).evaluation_service.get(evaluation_id).model_dump(mode="json")


@router.post("/tasks", status_code=201, response_model=TaskIdResponse)
def create_task(payload: TaskCreate, request: Request) -> dict[str, str]:
    return {"task_id": _service(request).create_task(payload)}


@router.get("/tasks", response_model=TaskListResponse)
def list_tasks(request: Request) -> dict[str, object]:
    return {"items": _service(request).list_tasks()}


@router.get("/tasks/{task_id}", response_model=TaskDetailResponse)
def task_detail(task_id: str, request: Request) -> dict[str, object]:
    return _service(request).task_detail(task_id)


@router.get("/tasks/{task_id}/history", response_model=HistoricalTaskResponse)
def historical_task(task_id: str, request: Request) -> dict[str, object]:
    return _service(request).historical_task(task_id)


@router.post("/tasks/{task_id}/runs", status_code=202, response_model=RunRefResponse)
async def start_run(
    task_id: str,
    payload: RunRequest,
    request: Request,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> dict[str, str]:
    reference = await _service(request).start_run(task_id, payload, _request_key(idempotency_key))
    return reference.model_dump()


@router.post(
    "/tasks/{task_id}/feedback",
    status_code=201,
    response_model=RevisionPlanResponse,
)
async def prepare_feedback(
    task_id: str,
    payload: FeedbackRequest,
    request: Request,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> dict[str, object]:
    service = _service(request)
    plan_id = await service.revision_service.prepare(
        task_id, payload, _request_key(idempotency_key)
    )
    return service.revision_service.get(plan_id).model_dump(mode="json")


@router.get(
    "/tasks/{task_id}/revisions/{plan_id}",
    response_model=RevisionPlanResponse,
)
def revision_plan(task_id: str, plan_id: str, request: Request) -> dict[str, object]:
    plan = _service(request).revision_service.get(plan_id)
    if plan.task_id != task_id:
        raise ServiceError("NOT_FOUND", "修订计划不存在。", status_code=404)
    return plan.model_dump(mode="json")


@router.post(
    "/tasks/{task_id}/revisions/{plan_id}/runs",
    status_code=202,
    response_model=RunRefResponse,
)
async def execute_revision(
    task_id: str,
    plan_id: str,
    payload: RevisionRunRequest,
    request: Request,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> dict[str, str]:
    service = _service(request)
    plan = service.revision_service.get(plan_id)
    if plan.task_id != task_id:
        raise ServiceError("NOT_FOUND", "修订计划不存在。", status_code=404)
    reference = await service.revision_service.execute(
        plan_id, payload.budget, _request_key(idempotency_key), strategy=payload.strategy, qd=payload.qd
    )
    return reference.model_dump()


@router.post("/runs/{run_id}/cancel", status_code=202, response_model=CancelRunResponse)
async def cancel_run(run_id: str, request: Request) -> dict[str, str]:
    status = await _service(request).cancel_run(run_id)
    return {"run_id": run_id, "status": status}


@router.post(
    "/tasks/{task_id}/runs/{run_id}/approvals/{request_id}",
    response_model=ApprovalDecisionResponse,
)
async def resolve_approval(
    task_id: str, run_id: str, request_id: str,
    payload: ApprovalDecisionRequest, request: Request,
) -> dict[str, str]:
    try:
        await request.app.state.run_manager.resolve_approval(
            task_id, run_id, request_id, payload.approved
        )
    except KeyError as error:
        raise ServiceError("NOT_FOUND", "审批请求不存在。", status_code=404) from error
    except ValueError as error:
        if "does not belong" in str(error):
            raise ServiceError("NOT_FOUND", "审批请求不存在。", status_code=404) from error
        raise ServiceError("APPROVAL_EXPIRED", "审批请求已失效。", status_code=409) from error
    return {"request_id": request_id, "decision": "approved" if payload.approved else "denied"}


@router.post("/runs/{run_id}/resume", status_code=202, response_model=RunRefResponse)
async def resume_run(
    run_id: str,
    request: Request,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> dict[str, str]:
    reference = await _service(request).resume_run(run_id, _request_key(idempotency_key))
    return reference.model_dump()


@router.get("/tasks/{task_id}/events")
async def task_events(
    task_id: str,
    request: Request,
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    after: int = 0,
) -> StreamingResponse:
    cursor = after
    if last_event_id is not None:
        try:
            cursor = max(cursor, int(last_event_id))
        except ValueError as error:
            raise ServiceError("INVALID_EVENT_CURSOR", "Last-Event-ID 必须是整数。") from error
    if cursor < 0:
        raise ServiceError("INVALID_EVENT_CURSOR", "事件游标不能为负数。")
    service = _service(request)
    # Validate the task before response headers are sent. Errors raised only
    # from the streaming generator would arrive too late to become an HTTP 404.
    service.events_after(task_id, cursor)

    async def body() -> AsyncIterator[str]:
        event_cursor = cursor
        last_heartbeat = 0.0
        while not await request.is_disconnected():
            events = service.events_after(task_id, event_cursor)
            if events:
                for event in events:
                    payload = json.dumps(
                        {
                            "task_id": event.task_id,
                            "run_id": event.run_id,
                            "seq": event.seq,
                            "type": event.type,
                            "payload": event.payload,
                            "created_at": event.created_at,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    yield f"id: {event.seq}\nevent: {event.type}\ndata: {payload}\n\n"
                    event_cursor = event.seq
                continue
            now = monotonic()
            if last_heartbeat == 0.0 or now - last_heartbeat >= _SSE_HEARTBEAT_SECONDS:
                yield ": heartbeat\n\n"
                last_heartbeat = now
            await asyncio.sleep(_SSE_POLL_SECONDS)

    return StreamingResponse(
        body(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/history/sessions", response_model=HistorySessionListResponse)
def history_sessions(request: Request) -> dict[str, object]:
    return {"items": _service(request).history_sessions()}


@router.post("/history/import", status_code=201, response_model=TaskIdResponse)
def import_history(payload: HistoryImportRequest, request: Request) -> dict[str, str]:
    return {"task_id": _service(request).import_history(payload.session_id)}


@router.post("/tasks/{task_id}/attachments", status_code=201, response_model=AttachmentResponse)
async def upload_attachment(
    task_id: str,
    request: Request,
    file: UploadFile = File(...),
) -> dict[str, object]:
    metadata = await _service(request).artifacts.save_upload(task_id, file)
    return {"attachment_id": metadata["artifact_id"], **metadata}


@router.get("/artifacts/{artifact_id}", response_model=ArtifactMetadataResponse)
def artifact_metadata(artifact_id: str, request: Request) -> dict[str, object]:
    return _service(request).artifacts.metadata(artifact_id)


def _artifact_chunks(content: tempfile.SpooledTemporaryFile[bytes]) -> Iterator[bytes]:
    while chunk := content.read(1024 * 1024):
        yield chunk


@router.get("/artifacts/{artifact_id}/content")
def artifact_content(artifact_id: str, request: Request) -> StreamingResponse:
    content, media_type, filename = _service(request).artifacts.content(artifact_id)
    disposition = "inline" if media_type.startswith("image/") else "attachment"
    return StreamingResponse(
        _artifact_chunks(content),
        media_type=media_type,
        headers={
            "Content-Disposition": f'{disposition}; filename="{filename}"',
            "X-Content-Type-Options": "nosniff",
        },
        background=BackgroundTask(content.close),
    )


@router.get("/capabilities", response_model=CapabilityListResponse)
def capabilities(request: Request) -> dict[str, object]:
    return {"items": _service(request).capability_list()}


__all__ = ["router"]


@router.get("/skills")
def list_skills(request: Request) -> dict[str, object]:
    return {"items": _service(request).skill_service.list_skills()}


@router.get("/skills/sources")
def skill_sources(request: Request) -> dict[str, object]:
    return {"items": _service(request).skill_sources()}


@router.post("/skills", status_code=201)
def create_skill(payload: SkillCreate, request: Request) -> dict[str, object]:
    service = _service(request)
    service.task_detail(payload.source.task_id)
    if payload.source.kind == "ordinary_run":
        if not any(run.run_id == payload.source.run_id and run.status in {"SUCCEEDED", "PARTIAL"} for run in service.store.runs_for_task(payload.source.task_id)):
            raise ServiceError("SKILL_SOURCE_INVALID", "源运行不属于该任务。")
    else:
        if not any(link.source_type == "qd_trial" and link.source_id == payload.source.trial_id for link in service.store.links_for_task(payload.source.task_id)):
            raise ServiceError("SKILL_QD_SOURCE_INVALID", "QD trial 不属于当前任务。")
    return service.skill_service.create(payload)


@router.post("/skills/{skill_id}/replay", status_code=202, response_model=RunRefResponse)
async def replay_skill(skill_id: str, payload: SkillReplay, request: Request,
                       idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")) -> dict[str, str]:
    reference = await _service(request).start_skill_replay(skill_id, payload, _request_key(idempotency_key))
    return reference.model_dump()


@router.post("/skills/{skill_id}/disable")
def disable_skill(skill_id: str, request: Request) -> dict[str, str]:
    return _service(request).skill_service.disable(skill_id)


@router.get("/procedures")
def list_procedures(request: Request) -> dict[str, object]:
    return {"items": ProcedureWorkflow(_service(request)).list_skills()}


@router.get("/tasks/{task_id}/revisions")
def list_revision_plans(task_id: str, request: Request) -> dict[str, object]:
    service = _service(request)
    service.task_detail(task_id)
    return {"items": [plan.model_dump(mode="json") for plan in service.store.revision_plans_for_task(task_id)]}


@router.post("/procedures", status_code=201)
def create_procedure(payload: ProcedureCreate, request: Request) -> dict[str, object]:
    return ProcedureWorkflow(_service(request)).create(payload)


@router.get("/procedures/{skill_id}")
def read_procedure(skill_id: str, request: Request) -> dict[str, object]:
    return ProcedureWorkflow(_service(request)).read(skill_id)


@router.post("/procedures/{skill_id}/approve")
def approve_procedure(skill_id: str, payload: ProcedureReview, request: Request) -> dict[str, object]:
    return ProcedureWorkflow(_service(request)).approve(skill_id, payload.review_note)


@router.post("/procedures/{skill_id}/disable")
def disable_procedure(skill_id: str, request: Request) -> dict[str, object]:
    return ProcedureWorkflow(_service(request)).disable(skill_id)
