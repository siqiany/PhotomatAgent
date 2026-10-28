"""FastAPI application factory with process-owned run-manager lifecycle."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from photomatagent.application.runner import RunManager
from photomatagent.application.errors import ServiceError
from photomatagent.application.service import WorkbenchService
from photomatagent.web.artifacts import MAX_UPLOAD_BYTES
from photomatagent.web.middleware import UploadBodyLimitMiddleware
from photomatagent.web.routes import router


def create_app(
    *,
    service: WorkbenchService,
    manager: RunManager,
    webui_directory: Path | str | None = None,
) -> FastAPI:
    """Create one local app; the lifespan owns the only execution manager."""

    if service.manager is not manager:
        raise ValueError("the app lifespan and service must share one RunManager")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            await manager.start()
            yield
        finally:
            try:
                await manager.stop()
            finally:
                service.store.close()

    app = FastAPI(title="PhotomatAgent 科研工作台", version="0.1.0", lifespan=lifespan)
    app.state.service = service
    app.state.run_manager = manager
    app.add_middleware(
        UploadBodyLimitMiddleware,
        max_body_bytes=MAX_UPLOAD_BYTES + 1024 * 1024,
    )
    app.include_router(router)

    static_directory = Path(webui_directory) if webui_directory is not None else None
    if static_directory is None:
        source_tree = Path(__file__).resolve().parents[3] / "webui" / "dist"
        packaged = Path(__file__).resolve().parent / "static"
        static_directory = next(
            (candidate for candidate in (source_tree, packaged) if (candidate / "index.html").is_file()),
            source_tree,
        )
    if (static_directory / "index.html").is_file():
        app.mount(
            "/",
            StaticFiles(directory=static_directory, html=True),
            name="workbench-webui",
        )

    @app.exception_handler(ServiceError)
    async def service_error_handler(_request: Request, error: ServiceError) -> JSONResponse:
        return JSONResponse(
            status_code=error.status_code,
            content={
                "error": {
                    "code": error.code,
                    "message": str(error),
                    "run_id": error.run_id,
                    "retryable": error.retryable,
                    "next_action": error.next_action,
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_request: Request, _error: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "INVALID_REQUEST",
                    "message": "请求字段缺失、格式错误或超出允许范围。",
                    "run_id": None,
                    "retryable": False,
                    "next_action": "检查输入字段后重试。",
                }
            },
        )

    @app.exception_handler(Exception)
    async def internal_error_handler(_request: Request, _error: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "服务暂时无法完成此操作。",
                    "run_id": None,
                    "retryable": True,
                    "next_action": "稍后重试；若问题持续，请查看本地服务日志。",
                }
            },
        )

    return app


__all__ = ["create_app"]
