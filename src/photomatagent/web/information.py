"""Project management and storage endpoints on the shared application service."""
from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Request, Query, Header
from pydantic import BaseModel

from photomatagent.application.projects import ProjectCreate, ProjectPatch, TaskPatch, RevisionRequest, MoveRequest, canonical, now
from photomatagent.application.errors import ServiceError

router = APIRouter(prefix="/api")


@router.get("/projects")
def projects(request: Request, archived: bool = False) -> dict[str, Any]:
    return {"items":request.app.state.service.store.projects.list(archived)}


@router.post("/projects", status_code=201)
def create_project(payload: ProjectCreate, request: Request) -> dict[str, Any]:
    return request.app.state.service.store.projects.create(payload)


@router.get("/projects/{project_id}")
def project(project_id: str, request: Request) -> dict[str, Any]:
    return request.app.state.service.store.projects.project(project_id)


@router.patch("/projects/{project_id}")
def patch_project(project_id: str, payload: ProjectPatch, request: Request) -> dict[str, Any]:
    return request.app.state.service.store.projects.patch_project(project_id,payload)


@router.post("/projects/{project_id}/{action}")
def project_action(project_id: str, action: str, payload: RevisionRequest, request: Request) -> dict[str, Any]:
    if action not in {"archive","unarchive"}:
        raise ServiceError("NOT_FOUND","操作不存在。",status_code=404)
    return request.app.state.service.store.projects.archive_project(project_id,payload.expected_revision,action=="archive")


@router.patch("/tasks/{task_id}")
def patch_task(task_id: str, payload: TaskPatch, request: Request) -> dict[str, Any]:
    return request.app.state.service.store.projects.edit_task(task_id,payload)


@router.post("/tasks/{task_id}/move")
def move_task(task_id: str, payload: MoveRequest, request: Request) -> dict[str, Any]:
    return request.app.state.service.store.projects.edit_task(task_id,revision=payload.expected_revision,project_id=payload.project_id,target_revision=payload.target_revision)


@router.post("/tasks/{task_id}/archive")
def archive_task(task_id: str, payload: RevisionRequest, request: Request) -> dict[str, Any]:
    return request.app.state.service.store.projects.edit_task(task_id,revision=payload.expected_revision,archived=True)


@router.post("/tasks/{task_id}/unarchive")
def unarchive_task(task_id: str, payload: RevisionRequest, request: Request) -> dict[str, Any]:
    return request.app.state.service.store.projects.edit_task(task_id,revision=payload.expected_revision,archived=False)


@router.get("/search")
def search(request: Request, q: str, project_id: str | None = None, limit: int = Query(30,ge=1,le=100)) -> dict[str, Any]:
    store = request.app.state.service.store
    result = store.projects.search(q, project_id=project_id, limit=limit)
    for item in result["items"]:
        if item["kind"] == "artifact":
            item["result_invalidation_ids"] = store.artifact_result_invalidations(
                item["document_id"].removeprefix("artifact:"),
            )
    return result


@router.get("/sessions/{session_id}/checkpoints")
def checkpoints(session_id: str, request: Request, limit: int = Query(50,ge=1,le=200)) -> dict[str, Any]:
    return {"items":request.app.state.service.storage.sessions.checkpoints(session_id,limit)}


@router.get("/checkpoints/{checkpoint_id}")
def checkpoint(checkpoint_id: str, request: Request) -> dict[str, Any]:
    store = request.app.state.service.store
    with store.database.lock:
        row = store._connection.execute("SELECT session_id FROM checkpoints WHERE checkpoint_id=?",(checkpoint_id,)).fetchone()
    if row is None:
        raise ServiceError("NOT_FOUND","检查点不存在。",status_code=404)
    snapshot = request.app.state.service.storage.sessions.load(row[0],checkpoint_id)
    return {"checkpoint_id":checkpoint_id,"session_id":row[0],"message_count":len(snapshot.conversation.messages),"scientific_goal":snapshot.scientific.goal,"authority":"historical; revalidation required"}


@router.get("/tasks/{task_id}/messages")
def messages(task_id: str, request: Request, after: int = Query(0,ge=0), limit: int = Query(50,ge=1,le=200)) -> dict[str, Any]:
    store = request.app.state.service.store
    task = store.task(task_id)
    with store.database.lock:
        rows = store._connection.execute("SELECT message_seq,payload_json FROM messages WHERE session_id=? AND message_seq>? ORDER BY message_seq LIMIT ?",(task.session_id,after,limit+1)).fetchall()
    return {"items":[{"seq":row[0],"message":json.loads(row[1])} for row in rows[:limit]],"next_after":rows[limit-1][0] if len(rows)>limit else None}


@router.get("/tasks/{task_id}/runs")
def runs(task_id: str, request: Request, before: str | None = None, limit: int = Query(50,ge=1,le=200)) -> dict[str, Any]:
    store = request.app.state.service.store
    store.task(task_id)
    with store.database.lock:
        boundary = store._connection.execute("SELECT created_at FROM runs WHERE task_id=? AND run_id=?",(task_id,before)).fetchone() if before else None
        if before and boundary is None:
            raise ServiceError("INVALID_CURSOR","运行游标不属于此任务。",status_code=422)
        rows = store._connection.execute("SELECT run_id FROM runs WHERE task_id=? AND (? IS NULL OR (created_at,run_id)<(?,?)) ORDER BY created_at DESC,run_id DESC LIMIT ?",(task_id,before,boundary[0] if boundary else None,before,limit+1)).fetchall()
    return {"items":[store.run(row[0]).model_dump(mode="json") for row in reversed(rows[:limit])],"next_before":rows[limit-1][0] if len(rows)>limit else None}


@router.get("/tasks/{task_id}/artifacts")
def artifacts(task_id: str, request: Request, before: str | None = None, limit: int = Query(50,ge=1,le=200)) -> dict[str, Any]:
    service = request.app.state.service
    service.store.task(task_id)
    with service.store.database.lock:
        conn = service.store._connection
        boundary = conn.execute("SELECT created_at FROM artifacts WHERE task_id=? AND artifact_id=?",(task_id,before)).fetchone() if before else None
        if before and boundary is None:
            raise ServiceError("INVALID_CURSOR","产物游标不属于此任务。",status_code=422)
        rows = conn.execute("SELECT * FROM artifacts WHERE task_id=? AND (? IS NULL OR (created_at,artifact_id)<(?,?)) ORDER BY created_at DESC,artifact_id DESC LIMIT ?",(task_id,before,boundary[0] if boundary else None,before,limit+1)).fetchall()
    return {"items":[service.artifacts.metadata_row(row) for row in rows[:limit]],"next_before":rows[limit-1]["artifact_id"] if len(rows)>limit else None}


class InputReference(BaseModel):
    source_artifact_id: str
    replaces_artifact_id: str | None = None


@router.post("/tasks/{task_id}/inputs")
def link_input(task_id: str, payload: InputReference, request: Request) -> dict[str, Any]:
    from photomatagent.persistence.maintenance import _sha
    service = request.app.state.service
    if payload.replaces_artifact_id:
        identifier = service.artifacts.replace_input(task_id, payload.replaces_artifact_id, payload.source_artifact_id)
        return {"artifact_id": identifier, "source_artifact_id": payload.source_artifact_id,
                "replaces_artifact_id": payload.replaces_artifact_id, "task_id": task_id}
    source = service.store.artifact(payload.source_artifact_id)
    path = service.storage.workspace.resolve(source["path"])
    if path.is_symlink() or path.stat().st_size!=source["size_bytes"] or _sha(path)!=source["sha256"]:
        raise ServiceError("SOURCE_CHANGED","输入来源无法通过完整性检查。",status_code=409)
    identifier = "artifact_"+uuid4().hex
    with service.store.database.transaction():
        service.store.projects._idle(task_id=task_id)
        if service.store.artifact_result_invalidations(source["artifact_id"]):
            raise ServiceError("INPUT_REVALIDATION_REQUIRED", "来源结果已失效，请使用重新验证后的产物。", status_code=409)
        service.store.register_attachment(artifact_id=identifier,task_id=task_id,path=source["path"],filename=source["filename"],sha256=source["sha256"],media_type=source["media_type"],size_bytes=source["size_bytes"])
        service.store._connection.execute("UPDATE artifacts SET origin_artifact_id=? WHERE artifact_id=?",(source["artifact_id"],identifier))
    return {"artifact_id":identifier,"source_artifact_id":source["artifact_id"],"task_id":task_id}


@router.get("/storage/status")
def storage_status(request: Request) -> dict[str, Any]:
    return request.app.state.service.storage.status()


@router.get("/storage/diagnostics")
def diagnostics(request: Request) -> dict[str, Any]:
    return request.app.state.service.storage.diagnostics()


class MaintenanceRequest(BaseModel):
    full: bool = False


@router.post("/storage/{action}")
def maintenance(action: str, payload: MaintenanceRequest, request: Request, idempotency_key: str = Header(alias="Idempotency-Key",min_length=1,max_length=200)) -> dict[str, Any]:
    service = request.app.state.service
    methods = {"import":service.storage.import_sessions,"refresh":service.storage.refresh,"reindex":service.storage.reindex,"export":service.storage.export,"backup":lambda:service.storage.backup(payload.full)}
    if action not in methods:
        raise ServiceError("NOT_FOUND","操作不存在。",status_code=404)
    if action in {"import","reindex"} and service.store.unfinished_runs():
        raise ServiceError("ACTIVE_RUN_EXISTS","运行结束后再执行导入或索引重建。",status_code=409)
    binding = canonical([action,payload.model_dump()])
    with service.store.database.transaction():
        existing = service.store._connection.execute("SELECT * FROM maintenance_operations WHERE request_key=?",(idempotency_key,)).fetchone()
        if existing:
            if existing["kind"]!=binding:
                raise ServiceError("IDEMPOTENCY_CONFLICT","请求键已用于其他操作。",status_code=409)
            return dict(existing) | {"result":json.loads(existing["result_json"]) if existing["result_json"] else None}
        identifier = "maintenance_"+uuid4().hex
        service.store._connection.execute("INSERT INTO maintenance_operations VALUES (?,?,?,'RUNNING',NULL,?)",(identifier,idempotency_key,binding,now()))
    try:
        result = methods[action]()
    except (RuntimeError,ValueError,OSError) as exc:
        with service.store.database.transaction():
            service.store._connection.execute("UPDATE maintenance_operations SET status='FAILED',result_json=? WHERE operation_id=?",(canonical({"code":type(exc).__name__,"diagnostic":str(exc)[:200]}),identifier))
        raise ServiceError("MAINTENANCE_FAILED","操作未完成；完整备份需停止工作台后离线执行。",status_code=409) from exc
    with service.store.database.transaction():
        service.store._connection.execute("UPDATE maintenance_operations SET status='SUCCEEDED',result_json=? WHERE operation_id=?",(canonical(result),identifier))
    return {"operation_id":identifier,"status":"SUCCEEDED","result":result}


@router.get("/storage/operations/{operation_id}")
def maintenance_status(operation_id: str, request: Request) -> dict[str, Any]:
    store = request.app.state.service.store
    with store.database.lock:
        row = store._connection.execute("SELECT * FROM maintenance_operations WHERE operation_id=?",(operation_id,)).fetchone()
    if row is None:
        raise ServiceError("NOT_FOUND","操作不存在。",status_code=404)
    return dict(row) | {"result":json.loads(row["result_json"]) if row["result_json"] else None}
