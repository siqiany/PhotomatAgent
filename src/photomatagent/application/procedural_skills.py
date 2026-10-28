"""Approved procedural guidance, kept separate from executable skills."""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


_TASK_ID = re.compile(r"^task_[0-9a-f]{32}$")
_RUN_ID = re.compile(r"^run_[0-9a-f]{32}$")
_SKILL_ID = re.compile(r"^procedure_[0-9a-f]{32}$")


class ProcedureCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9-]{0,59}$")
    description: str = Field(min_length=1, max_length=500)
    scope: str = Field(min_length=1, max_length=200)
    instructions: str = Field(min_length=1, max_length=20_000)
    task_id: str = Field(pattern=r"^task_[0-9a-f]{32}$")
    run_id: str = Field(pattern=r"^run_[0-9a-f]{32}$")
    plan_id: str | None = Field(default=None, min_length=1, max_length=128)
    artifact_id: str | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def source_required(self) -> ProcedureCreate:
        if self.plan_id is None and self.artifact_id is None:
            raise ValueError("PROCEDURE_SOURCE_REQUIRED")
        return self


class ProcedureService:
    """Store procedural notes with integrity checks and explicit review records."""

    def __init__(self, workspace: Path | str) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.root = self.workspace / ".photomatagent" / "procedural-skills"

    def create(self, payload: ProcedureCreate | dict[str, Any], *, provenance: dict[str, Any]) -> dict[str, Any]:
        request = payload if isinstance(payload, ProcedureCreate) else ProcedureCreate.model_validate(payload)
        if not isinstance(provenance, dict) or not provenance:
            raise ValueError("PROCEDURE_PROVENANCE_REQUIRED")
        self._ensure_root()
        skill_id = f"procedure_{uuid4().hex}"
        index_name = f"procedure-{request.name}-{uuid4().hex[:8]}"
        directory = self.root / skill_id
        directory.mkdir(mode=0o700)
        content = _render_skill(request, index_name)
        candidate = {
            "schema_version": 1,
            "skill_id": skill_id,
            "index_name": index_name,
            "name": request.name,
            "description": request.description,
            "scope": request.scope,
            "instructions": request.instructions,
            "task_id": request.task_id,
            "run_id": request.run_id,
            "plan_id": request.plan_id,
            "artifact_id": request.artifact_id,
            "provenance": provenance,
            "status": "CANDIDATE",
            "validation_status": "UNKNOWN",
            "execution_status": "NOT_EXECUTABLE",
            "skill_sha256": _sha256_bytes(content.encode("utf-8")),
        }
        candidate_bytes = _json_bytes(candidate)
        candidate_sha256 = _sha256_bytes(candidate_bytes)
        self._write_new(directory / "SKILL.md", content.encode("utf-8"))
        self._write_new(directory / "candidate.json", candidate_bytes)
        self._write_new(directory / "state.json", _json_bytes({
            "status": "CANDIDATE", "candidate_sha256": candidate_sha256,
            "updated_at": _now(), "review_note": None,
        }))
        return self._public(candidate, candidate_sha256, "CANDIDATE", content)

    def list_skills(self) -> list[dict[str, Any]]:
        if not self._safe_root_exists():
            return []
        result = []
        for directory in sorted(self.root.iterdir()):
            if not _SKILL_ID.fullmatch(directory.name) or directory.is_symlink() or not directory.is_dir():
                continue
            try:
                record = self._read_record(directory.name)
                result.append({key: value for key, value in record.items() if key != "instructions" and key != "skill_content"})
            except (OSError, ValueError, TypeError, KeyError):
                continue
        return result

    def read(self, skill_id: str, *, task_id: str | None = None, run_id: str | None = None) -> dict[str, Any]:
        if task_id is not None and not _TASK_ID.fullmatch(task_id):
            raise ValueError("PROCEDURE_TASK_ID_INVALID")
        if run_id is not None and not _RUN_ID.fullmatch(run_id):
            raise ValueError("PROCEDURE_RUN_ID_INVALID")
        record = self._read_record(skill_id)
        if task_id is not None or run_id is not None:
            self.record_retrieval(skill_id, task_id=task_id, run_id=run_id)
        return record

    def approve(self, skill_id: str, review_note: str) -> dict[str, Any]:
        note = review_note.strip() if isinstance(review_note, str) else ""
        if not note:
            raise ValueError("PROCEDURE_REVIEW_NOTE_REQUIRED")
        if len(note) > 2000:
            raise ValueError("PROCEDURE_REVIEW_NOTE_TOO_LONG")
        record = self._read_record(skill_id)
        if record["status"] == "DISABLED":
            raise ValueError("PROCEDURE_DISABLED")
        self._write_state(skill_id, record["candidate_sha256"], "APPROVED_PROCEDURE", note)
        return self._read_record(skill_id)

    def disable(self, skill_id: str) -> dict[str, Any]:
        record = self._read_record(skill_id)
        self._write_state(skill_id, record["candidate_sha256"], "DISABLED", None)
        return self._read_record(skill_id)

    def record_retrieval(
        self, skill_id: str, *, task_id: str | None = None, run_id: str | None = None,
    ) -> dict[str, Any]:
        if task_id is not None and not _TASK_ID.fullmatch(task_id):
            raise ValueError("PROCEDURE_TASK_ID_INVALID")
        if run_id is not None and not _RUN_ID.fullmatch(run_id):
            raise ValueError("PROCEDURE_RUN_ID_INVALID")
        record = self._read_record(skill_id)
        retrieval = {
            "skill_id": skill_id, "task_id": task_id, "run_id": run_id,
            "retrieved_at": _now(), "status": "GUIDANCE_RETRIEVED",
        }
        retrieval["retrieval_sha256"] = _sha256_bytes(_json_bytes(retrieval))
        path = self.root / "retrievals.jsonl"
        self._append_jsonl(path, retrieval)
        return retrieval

    def list_retrievals(self, skill_id: str | None = None) -> list[dict[str, Any]]:
        path = self.root / "retrievals.jsonl"
        if not self._safe_root_exists() or path.is_symlink() or not path.is_file():
            return []
        result = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
                digest = record.pop("retrieval_sha256")
                if _sha256_bytes(_json_bytes(record)) != digest:
                    continue
                record["retrieval_sha256"] = digest
                if skill_id is None or record["skill_id"] == skill_id:
                    result.append(record)
            except (ValueError, TypeError, KeyError, AttributeError):
                continue
        return result

    def _read_record(self, skill_id: str) -> dict[str, Any]:
        if not _SKILL_ID.fullmatch(skill_id):
            raise ValueError("PROCEDURE_ID_INVALID")
        directory = self.root / skill_id
        self._assert_directory(directory)
        candidate_path, content_path, state_path = (
            directory / "candidate.json", directory / "SKILL.md", directory / "state.json"
        )
        for path in (candidate_path, content_path, state_path):
            if path.is_symlink() or not path.is_file():
                raise ValueError("PROCEDURE_PATH_INVALID")
        candidate_bytes = candidate_path.read_bytes()
        candidate_sha256 = _sha256_bytes(candidate_bytes)
        try:
            candidate = json.loads(candidate_bytes)
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("PROCEDURE_RECORD_CHANGED") from exc
        if not isinstance(candidate, dict) or not isinstance(state, dict):
            raise ValueError("PROCEDURE_RECORD_CHANGED")
        if candidate.get("skill_id") != skill_id:
            raise ValueError("PROCEDURE_RECORD_CHANGED")
        content = content_path.read_text(encoding="utf-8")
        if _sha256_bytes(content.encode("utf-8")) != candidate.get("skill_sha256"):
            raise ValueError("PROCEDURE_CONTENT_CHANGED")
        if state.get("candidate_sha256") != candidate_sha256:
            raise ValueError("PROCEDURE_RECORD_CHANGED")
        if state.get("status") not in {"CANDIDATE", "APPROVED_PROCEDURE", "DISABLED"}:
            raise ValueError("PROCEDURE_STATE_INVALID")
        return self._public(candidate, candidate_sha256, state["status"], content, state.get("review_note"))

    def _public(
        self, candidate: dict[str, Any], candidate_sha256: str, status: str,
        content: str, review_note: str | None = None,
    ) -> dict[str, Any]:
        return {
            **candidate,
            "status": status,
            "candidate_sha256": candidate_sha256,
            "review_note": review_note,
            "skill_content": content,
        }

    def _write_state(self, skill_id: str, candidate_sha256: str, status: str, note: str | None) -> None:
        directory = self.root / skill_id
        self._assert_directory(directory)
        path = directory / "state.json"
        if path.is_symlink() or not path.is_file():
            raise ValueError("PROCEDURE_PATH_INVALID")
        temporary = directory / f".state-{uuid4().hex}.tmp"
        self._write_new(temporary, _json_bytes({
            "status": status, "candidate_sha256": candidate_sha256,
            "updated_at": _now(), "review_note": note,
        }))
        os.replace(temporary, path)

    def _ensure_root(self) -> None:
        current = self.workspace
        for part in (".photomatagent", "procedural-skills"):
            current = current / part
            if current.is_symlink():
                raise ValueError("PROCEDURE_PATH_INVALID")
            current.mkdir(mode=0o700, exist_ok=True)
            if not current.is_dir():
                raise ValueError("PROCEDURE_PATH_INVALID")

    def _safe_root_exists(self) -> bool:
        return not self.root.is_symlink() and self.root.is_dir() and self.root.resolve() == self.root

    def _assert_directory(self, directory: Path) -> None:
        if (
            not self._safe_root_exists() or directory.is_symlink() or not directory.is_dir()
            or directory.resolve().parent != self.root
        ):
            raise ValueError("PROCEDURE_PATH_INVALID")

    @staticmethod
    def _write_new(path: Path, data: bytes) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def _render_skill(request: ProcedureCreate, index_name: str) -> str:
    return (
        "---\n"
        f"name: {index_name}\n"
        f"description: Procedural guidance for {request.scope[:160]}; not validated science.\n"
        "category: procedural-guidance\n"
        "tags: local, procedure\n"
        "---\n\n"
        f"# {request.description}\n\n"
        f"Scope: {request.scope}\n\n"
        "Procedural guidance drafted from prior work; approval status is recorded separately. "
        "It is not scientific validation, a checker PASS, or an executable skill. "
        "Validation state: UNKNOWN (UNVALIDATED). Reassess each recommendation against current "
        "evidence and record limitations.\n\n"
        "## Instructions\n\n"
        f"{request.instructions.strip()}\n\n"
        "## Limitations\n\n"
        "This guidance has not been validated as a scientific conclusion. Retrieval does not "
        "indicate successful execution or scientific correctness. No scripts are attached.\n"
    )


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
