"""Request-body limits applied before FastAPI parses multipart uploads."""

from __future__ import annotations

import json

from starlette.types import ASGIApp, Message, Receive, Scope, Send


class _RequestTooLarge(Exception):
    pass


class UploadBodyLimitMiddleware:
    """Bound multipart ingress before Starlette spools UploadFile content."""

    def __init__(self, app: ASGIApp, *, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = str(scope.get("path", ""))
        is_upload = (
            scope.get("type") == "http"
            and scope.get("method") == "POST"
            and path.startswith("/api/tasks/")
            and path.endswith("/attachments")
        )
        if not is_upload:
            await self.app(scope, receive, send)
            return

        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        raw_length = headers.get(b"content-length")
        if raw_length is not None:
            try:
                if int(raw_length) > self.max_body_bytes:
                    await self._too_large(send)
                    return
            except ValueError:
                pass

        total = 0

        async def limited_receive() -> Message:
            nonlocal total
            message = await receive()
            if message.get("type") == "http.request":
                total += len(message.get("body", b""))
                if total > self.max_body_bytes:
                    raise _RequestTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _RequestTooLarge:
            await self._too_large(send)

    @staticmethod
    async def _too_large(send: Send) -> None:
        body = json.dumps(
            {
                "error": {
                    "code": "UPLOAD_TOO_LARGE",
                    "message": "上传请求体超过允许的 51 MiB（含 multipart 开销）。",
                    "run_id": None,
                    "retryable": False,
                    "next_action": "缩小单个文件后重试。",
                }
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [(b"content-type", b"application/json; charset=utf-8")],
            }
        )
        await send({"type": "http.response.body", "body": body})


__all__ = ["UploadBodyLimitMiddleware"]
