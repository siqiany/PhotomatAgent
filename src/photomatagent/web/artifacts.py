"""Task-scoped upload storage and bounded artifact reads."""

from __future__ import annotations

import hashlib
import codecs
import os
from pathlib import Path
import re
import tempfile
import uuid

from fastapi import UploadFile

from photomatagent.application.errors import ServiceError
from photomatagent.application.store import WorkbenchStore

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024
_ALLOWED_SUFFIXES = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".csv": "text/csv",
    ".json": "application/json",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


class ArtifactService:
    def __init__(
        self,
        store: WorkbenchStore,
        workspace_root: Path | str,
        *,
        max_upload_bytes: int = MAX_UPLOAD_BYTES,
        max_download_bytes: int = MAX_DOWNLOAD_BYTES,
    ) -> None:
        self.store = store
        self.workspace_root = Path(workspace_root).resolve()
        self.upload_root = (self.workspace_root / ".photomatagent" / "workbench" / "uploads").resolve()
        self.max_upload_bytes = max_upload_bytes
        self.max_download_bytes = max_download_bytes

    async def save_upload(self, task_id: str, upload: UploadFile) -> dict[str, object]:
        try:
            self.store.task(task_id)
        except KeyError as error:
            raise ServiceError("NOT_FOUND", "任务不存在。", status_code=404) from error

        filename = Path(upload.filename or "upload").name
        suffix = Path(filename).suffix.lower()
        allowed_media_type = _ALLOWED_SUFFIXES.get(suffix)
        if allowed_media_type is None:
            raise ServiceError("UNSUPPORTED_MEDIA_TYPE", "仅支持 TXT、Markdown、CSV、JSON、PNG 和 JPEG。", status_code=415)
        media_type = allowed_media_type
        artifact_id = f"artifact_{uuid.uuid4().hex}"
        task_dir = self.upload_root / task_id
        try:
            task_dir.relative_to(self.upload_root)
        except ValueError as error:
            raise ServiceError("INVALID_TASK_PATH", "任务上传路径越界。", status_code=400) from error
        task_dir.mkdir(parents=True, exist_ok=True)
        resolved_task_dir = task_dir.resolve()
        if resolved_task_dir != task_dir:
            raise ServiceError("INVALID_TASK_PATH", "任务上传目录不能是符号链接。", status_code=400)

        digest = hashlib.sha256()
        size = 0
        temporary: Path | None = None
        final_path: Path | None = None
        decoder = codecs.getincrementaldecoder("utf-8")() if suffix in {".txt", ".md", ".csv", ".json"} else None
        try:
            with tempfile.NamedTemporaryFile(dir=task_dir, prefix=".upload-", delete=False) as output:
                temporary = Path(output.name)
                while chunk := await upload.read(1024 * 1024):
                    size += len(chunk)
                    if size > self.max_upload_bytes:
                        raise ServiceError("UPLOAD_TOO_LARGE", "单个文件不能超过 50 MiB。", status_code=413)
                    if decoder is not None:
                        try:
                            decoder.decode(chunk)
                        except UnicodeDecodeError as error:
                            raise ServiceError("INVALID_TEXT_ENCODING", "文本文件必须使用 UTF-8 编码。", status_code=422) from error
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            if decoder is not None:
                try:
                    decoder.decode(b"", final=True)
                except UnicodeDecodeError as error:
                    raise ServiceError("INVALID_TEXT_ENCODING", "文本文件必须使用 UTF-8 编码。", status_code=422) from error
            if suffix == ".png":
                with temporary.open("rb") as image:
                    if image.read(8) != b"\x89PNG\r\n\x1a\n":
                        raise ServiceError("INVALID_IMAGE", "PNG 文件签名无效。", status_code=422)
            elif suffix in {".jpg", ".jpeg"}:
                with temporary.open("rb") as image:
                    if image.read(3) != b"\xff\xd8\xff":
                        raise ServiceError("INVALID_IMAGE", "JPEG 文件签名无效。", status_code=422)
            final_path = task_dir / f"{artifact_id}{suffix}"
            os.replace(temporary, final_path)
            temporary = None
            relative_path = final_path.relative_to(self.workspace_root).as_posix()
            self.store.register_attachment(
                artifact_id=artifact_id,
                task_id=task_id,
                path=relative_path,
                filename=filename[:255],
                sha256=digest.hexdigest(),
                media_type=media_type,
                size_bytes=size,
            )
            return self.metadata_row(self.store.artifact(artifact_id))
        except BaseException:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            if final_path is not None and final_path.exists():
                final_path.unlink(missing_ok=True)
            raise
        finally:
            await upload.close()

    def metadata(self, artifact_id: str) -> dict[str, object]:
        try:
            return self.metadata_row(self.store.artifact(artifact_id))
        except KeyError as error:
            raise ServiceError("NOT_FOUND", "产物不存在。", status_code=404) from error

    def metadata_row(self, row: object) -> dict[str, object]:
        return {
            "artifact_id": row["artifact_id"],  # type: ignore[index]
            "task_id": row["task_id"],  # type: ignore[index]
            "run_id": row["run_id"],  # type: ignore[index]
            "filename": row["filename"],  # type: ignore[index]
            "media_type": row["media_type"],  # type: ignore[index]
            "size_bytes": row["size_bytes"],  # type: ignore[index]
            "sha256": row["sha256"],  # type: ignore[index]
        }

    def content(self, artifact_id: str) -> tuple[tempfile.SpooledTemporaryFile[bytes], str, str]:
        try:
            row = self.store.artifact(artifact_id)
        except KeyError as error:
            raise ServiceError("NOT_FOUND", "产物不存在。", status_code=404) from error
        if row["media_type"] not in set(_ALLOWED_SUFFIXES.values()):
            raise ServiceError("UNSUPPORTED_MEDIA_TYPE", "此类文件不允许通过工作台下载。", status_code=415)
        if int(row["size_bytes"]) > self.max_download_bytes:
            raise ServiceError("ARTIFACT_TOO_LARGE", "此产物超过下载大小上限。", status_code=413)
        path = (self.workspace_root / str(row["path"])).resolve()
        task_id = str(row["task_id"])
        allowed_dirs: tuple[Path, ...]
        if row["run_id"] is None:
            allowed_dirs = (self.upload_root / task_id,)
        else:
            allowed_dirs = (
                self.workspace_root / ".photomatagent" / "workbench" / "artifacts"
                / task_id / str(row["run_id"]),
                self.workspace_root / "user_output" / task_id,
            )
        if not any(
            allowed_dir.resolve() == allowed_dir and path.is_relative_to(allowed_dir)
            for allowed_dir in allowed_dirs
        ):
            raise ServiceError("INVALID_ARTIFACT_PATH", "产物路径不在工作台上传目录中。", status_code=404)
        if not path.is_file():
            raise ServiceError("ARTIFACT_UNAVAILABLE", "产物文件缺失或已变化。", status_code=404)
        snapshot = tempfile.SpooledTemporaryFile(max_size=1024 * 1024, mode="w+b")
        digest = hashlib.sha256()
        size = 0
        try:
            # Copy and verify the exact bytes that will be streamed. This avoids
            # a check-then-reopen race if a running agent rewrites its output.
            with path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    size += len(chunk)
                    if size > self.max_download_bytes:
                        raise ServiceError("ARTIFACT_TOO_LARGE", "此产物超过下载大小上限。", status_code=413)
                    digest.update(chunk)
                    snapshot.write(chunk)
            if size != int(row["size_bytes"]):
                raise ServiceError("ARTIFACT_UNAVAILABLE", "产物文件缺失或已变化。", status_code=404)
            if digest.hexdigest() != row["sha256"]:
                raise ServiceError("ARTIFACT_INTEGRITY_ERROR", "产物摘要校验失败。", status_code=409)
            snapshot.seek(0)
        except BaseException:
            snapshot.close()
            raise
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", str(row["filename"]))[:120] or "artifact"
        return snapshot, str(row["media_type"]), safe_name


__all__ = ["ArtifactService", "MAX_DOWNLOAD_BYTES", "MAX_UPLOAD_BYTES"]
