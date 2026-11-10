"""Task-scoped upload storage and bounded artifact reads."""

from __future__ import annotations

import hashlib
import codecs
import os
from pathlib import Path
import re
import shutil
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
    ".webp": "image/webp",
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
            raise ServiceError("UNSUPPORTED_MEDIA_TYPE", "仅支持 TXT、Markdown、CSV、JSON、PNG、JPEG 和 WebP。", status_code=415)
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
            elif suffix == ".webp":
                with temporary.open("rb") as image:
                    header = image.read(12)
                    if header[:4] != b"RIFF" or header[8:] != b"WEBP":
                        raise ServiceError("INVALID_IMAGE", "WebP 文件签名无效。", status_code=422)
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
        task_id: str = row["task_id"]  # type: ignore[index]
        run_id: str | None = row["run_id"]  # type: ignore[index]
        return {
            "artifact_id": row["artifact_id"],  # type: ignore[index]
            "task_id": task_id,
            "run_id": run_id,
            "result_invalidation_ids": self.store.artifact_result_invalidations(row["artifact_id"]),  # type: ignore[index]
            "filename": row["filename"],  # type: ignore[index]
            "media_type": row["media_type"],  # type: ignore[index]
            "size_bytes": row["size_bytes"],  # type: ignore[index]
            "sha256": row["sha256"],  # type: ignore[index]
        }

    def replace_input(self, task_id: str, previous_id: str, source_id: str) -> str:
        """Bind a fresh upstream result to a new, hash-verified local input."""
        destination: Path | None = None
        try:
            with self.store.database.transaction():
                task = self.store.task(task_id)
                prior = self.store._connection.execute(
                    "SELECT json_extract(payload_json, '$.replacement_artifact_id'), "
                    "json_extract(payload_json, '$.source_artifact_id') FROM management_events "
                    "WHERE kind='task_input_replaced' AND entity_id=? "
                    "AND json_extract(payload_json, '$.previous_artifact_id')=?", (task_id, previous_id),
                ).fetchone()
                if prior:
                    if prior[1] != source_id:
                        raise ServiceError("REPLACEMENT_INPUT_CHANGED", "该输入已绑定到另一份替换产物。", status_code=409)
                    return str(prior[0])
                self.store.projects._idle(task_id=task_id)
                previous = self.store.artifact(previous_id)
                source = self.store.artifact(source_id)
                if (previous["task_id"] != task_id or previous["run_id"] is not None
                        or previous["origin_artifact_id"] is None or previous_id not in task.attachment_ids
                        or not self.store.artifact_result_invalidations(previous_id)):
                    raise ServiceError("REPLACEMENT_INPUT_NOT_STALE", "请选择当前任务中已失效的关联输入。", status_code=409)
                origin = previous
                visited = {previous_id}
                while origin["origin_artifact_id"] is not None:
                    origin_id = origin["origin_artifact_id"]
                    if origin_id in visited:
                        raise ServiceError("REPLACEMENT_ORIGIN_INVALID", "输入来源存在循环。", status_code=409)
                    visited.add(origin_id)
                    origin = self.store.artifact(origin_id)
                if (source_id in visited or source["task_id"] == task_id
                        or source["task_id"] != origin["task_id"] or source["media_type"] != previous["media_type"]
                        or source["run_id"] is None or self.store.run(source["run_id"]).status != "SUCCEEDED"
                        or self.store.artifact_result_invalidations(source_id)):
                    raise ServiceError("REPLACEMENT_SOURCE_INVALID", "请选择原上游任务成功运行产生的当前同类产物。", status_code=409)
                snapshot, media, _ = self.content(source_id)
                identifier = "artifact_" + uuid.uuid4().hex
                directory = self.upload_root / task_id
                with snapshot:
                    if self.upload_root != self.workspace_root / ".photomatagent" / "workbench" / "uploads":
                        raise ServiceError("INVALID_TASK_PATH", "上传根目录不能是符号链接。", status_code=400)
                    directory.mkdir(parents=True, exist_ok=True)
                    if directory.resolve() != directory:
                        raise ServiceError("INVALID_TASK_PATH", "任务上传目录不能是符号链接。", status_code=400)
                    path = directory / (identifier + Path(source["filename"]).suffix.lower())
                    with path.open("xb") as out:
                        destination = path
                        shutil.copyfileobj(snapshot, out)
                        out.flush()
                        os.fsync(out.fileno())
                self.store.register_attachment(
                    artifact_id=identifier, task_id=task_id, path=destination.relative_to(self.workspace_root).as_posix(),
                    filename=source["filename"], sha256=source["sha256"], media_type=media, size_bytes=source["size_bytes"],
                )
                self.store._connection.execute("UPDATE artifacts SET origin_artifact_id=? WHERE artifact_id=?", (source_id, identifier))
                self.store.bind_replacement_input(task_id, previous_id, identifier, source_id)
                return identifier
        except BaseException as exc:
            if destination is not None:
                destination.unlink(missing_ok=True)
            if isinstance(exc, KeyError):
                raise ServiceError("NOT_FOUND", "任务或产物不存在。", status_code=404) from exc
            raise

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
