"""Bind explicit workbench uploads to model image references."""

from __future__ import annotations

import hashlib

from photomatagent.application.errors import ServiceError
from photomatagent.application.store import WorkbenchStore
from photomatagent.errors import ToolExecutionError
from photomatagent.models.image_assets import MAX_IMAGE_BYTES, register_image_bytes
from photomatagent.models.types import ImageReference
from photomatagent.workspace import Workspace

_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp"}


def bind_uploaded_images(
    store: WorkbenchStore, workspace: Workspace, task_id: str, attachment_ids: list[str],
) -> tuple[ImageReference, ...]:
    result = []
    for attachment_id in attachment_ids:
        try:
            row = store.artifact(attachment_id)
        except KeyError as exc:
            raise ServiceError("IMAGE_ATTACHMENT_UNAVAILABLE", "图像附件不存在。", status_code=404) from exc
        if row["task_id"] != task_id or row["run_id"] is not None:
            raise ServiceError("IMAGE_ATTACHMENT_UNAVAILABLE", "图像附件不属于当前任务。", status_code=404)
        if row["media_type"] not in _IMAGE_TYPES or not 0 < row["size_bytes"] <= MAX_IMAGE_BYTES:
            raise ServiceError("INVALID_IMAGE_ATTACHMENT", "请选择不超过 4 MiB 的图像附件。", status_code=422)
        try:
            relative = str(row["path"])
            path = workspace.resolve(relative)
            if (workspace.relative(path) != relative or not path.is_file()
                    or path.parent != workspace.root / ".photomatagent" / "workbench" / "uploads" / task_id):
                raise ToolExecutionError("INVALID_IMAGE_PATH")
            with path.open("rb") as handle:
                data = handle.read(MAX_IMAGE_BYTES + 1)
            if len(data) != row["size_bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ToolExecutionError("IMAGE_ATTACHMENT_CHANGED")
            image = register_image_bytes(workspace, data)
            if image.media_type != row["media_type"]:
                raise ToolExecutionError("IMAGE_MEDIA_TYPE_CHANGED")
        except (OSError, ToolExecutionError, ImportError) as exc:
            raise ServiceError("INVALID_IMAGE_ATTACHMENT", "图像无法校验或读取。", status_code=422) from exc
        result.append(image)
    return tuple(result)
