"""Local, same-origin model configuration; no credential read endpoint."""
from urllib.parse import urlsplit

from fastapi import APIRouter, Request

from photomatagent.application.errors import ServiceError
from photomatagent.model_settings import ModelRole, ModelSettingsUpdate, public_settings, save_settings

router = APIRouter(prefix="/api/settings/models")


def _local_request(request: Request) -> None:
    # This local workbench has no account authentication. Credential administration
    # must not be exposed to network peers or cross-origin browser requests.
    client = request.client.host if request.client else ""
    host = request.url.hostname
    if client not in ("127.0.0.1", "::1", "localhost", "testclient") or host not in ("127.0.0.1", "::1", "localhost", "testserver"):
        raise ServiceError("LOCAL_SETTINGS_ONLY", "模型配置仅允许本机访问。", status_code=403)
    origin = request.headers.get("origin")
    if origin:
        parsed = urlsplit(origin)
        if parsed.scheme != request.url.scheme or parsed.netloc != request.url.netloc:
            raise ServiceError("CROSS_ORIGIN_SETTINGS", "不允许跨来源修改模型配置。", status_code=403)
    if request.headers.get("sec-fetch-site") == "cross-site":
        raise ServiceError("CROSS_ORIGIN_SETTINGS", "不允许跨来源访问模型配置。", status_code=403)


@router.get("")
def get_settings(request: Request) -> dict[str, dict[str, object]]:
    _local_request(request)
    return public_settings(request.app.state.service.workspace_root)


@router.put("/{role}")
async def update_settings(role: ModelRole, payload: ModelSettingsUpdate, request: Request) -> dict[str, dict[str, object]]:
    _local_request(request)
    service = request.app.state.service
    try:
        save_settings(service.workspace_root, role, payload)
    except ValueError as exc:
        raise ServiceError("INVALID_MODEL_SETTINGS", str(exc), status_code=422) from exc
    if role == "main":
        service.manager.configure_model(payload.provider, payload.model.strip())
    return public_settings(service.workspace_root)
