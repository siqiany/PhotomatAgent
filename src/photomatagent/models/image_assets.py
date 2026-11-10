"""Content-addressed local images for a model request."""

from __future__ import annotations

import base64
import hashlib
from io import BytesIO
from pathlib import Path
from typing import Literal

from photomatagent.errors import ToolExecutionError
from photomatagent.models.types import ImageReference, ModelMessage, active_image_entries
from photomatagent.persistence.atomic import atomic_write
from photomatagent.workspace import Workspace

MAX_IMAGE_BYTES = 4 * 1024 * 1024
_MEDIA: dict[str, Literal["image/png", "image/jpeg", "image/webp"]] = {
    "PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp",
}


def _asset_path(workspace: Workspace, digest: str) -> Path:
    return workspace.resolve(f".photomatagent/model-assets/{digest}", must_exist=False)


def register_image(workspace: Workspace, name: str) -> ImageReference:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or relative.parts[:1] not in {
        ("papers",), ("user_input",), ("user_output",), ("tmp",),
    }:
        raise ToolExecutionError("Image path must be relative to an approved workspace folder.")
    source = workspace.resolve(name)
    if workspace.relative(source).split("/", 1)[0] != relative.parts[0] or not source.is_file():
        raise ToolExecutionError("Image path does not identify an approved local file.")
    with source.open("rb") as handle:
        data = handle.read(MAX_IMAGE_BYTES + 1)
    return register_image_bytes(workspace, data)


def register_image_bytes(workspace: Workspace, data: bytes) -> ImageReference:
    from PIL import Image, UnidentifiedImageError

    if len(data) > MAX_IMAGE_BYTES:
        raise ToolExecutionError("IMAGE_TOO_LARGE")
    try:
        with Image.open(BytesIO(data)) as image:
            media = _MEDIA.get(image.format or "")
            if not media or max(image.size) > 4096 or getattr(image, "n_frames", 1) != 1:
                raise ToolExecutionError("IMAGE_FORMAT_OR_DIMENSIONS_UNSUPPORTED")
            image.verify()
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        raise ToolExecutionError("IMAGE_INVALID") from exc
    digest = hashlib.sha256(data).hexdigest()
    atomic_write(_asset_path(workspace, digest), data)
    return ImageReference(sha256=digest, media_type=media, byte_size=len(data))


def load_latest_images(workspace: Workspace, messages: list[ModelMessage]) -> dict[str, str]:
    encoded = {}
    for images in active_image_entries(messages).values():
        for image in images:
            path = _asset_path(workspace, image.sha256)
            if path.is_file():
                with path.open("rb") as handle:
                    data = handle.read(MAX_IMAGE_BYTES + 1)
                if len(data) == image.byte_size and hashlib.sha256(data).hexdigest() == image.sha256:
                    encoded[image.sha256] = base64.b64encode(data).decode("ascii")
    return encoded
