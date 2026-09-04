"""Deterministic primary-result materialization for evolution episodes."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from photomatagent.errors import ToolExecutionError
from photomatagent.models.types import AssistantMessage
from photomatagent.runtime.events import (
    RuntimeEvent,
    ToolCallCompleted,
    ToolCompleted,
    ToolFailed,
)
from photomatagent.runtime.state import ConversationState
from photomatagent.scientific.evolution.models import (
    ArtifactRef,
    validate_managed_id,
)
from photomatagent.workspace import Workspace

_EPISODE_VERSION = re.compile(r"^v[0-9]{3}$")
_RESULT_NAME = "result.md"
_WRITE_TOOLS = frozenset({"write", "edit"})
_CallIdentity = tuple[str | None, str | None, int, str]


class EpisodeArtifactError(RuntimeError):
    """Base error for deterministic episode-result selection."""


class MissingEpisodeResultError(EpisodeArtifactError):
    """Raised when neither a registered result nor assistant fallback exists."""


class EpisodeResultAlreadyExistsError(EpisodeArtifactError):
    """Raised when fallback materialization would replace an existing path."""


@dataclass(frozen=True, slots=True)
class _PendingWrite:
    tool_name: str
    path: str


@dataclass(slots=True)
class EpisodeArtifactCollector:
    """Correlate declared write/edit calls with their successful completion.

    The collector records event facts only. It deliberately does not inspect the
    filesystem; selection and containment checks happen later against the
    executor's authoritative :class:`Workspace`.
    """

    _pending: dict[_CallIdentity, _PendingWrite] = field(default_factory=dict)
    _terminal: set[_CallIdentity] = field(default_factory=set)
    _successful_paths: list[str] = field(default_factory=list)

    def observe(self, event: RuntimeEvent) -> None:
        identity = self._identity(event)
        if identity is None:
            return
        if isinstance(event, ToolCallCompleted):
            self._observe_call(identity, event)
            return
        if isinstance(event, ToolFailed):
            self._pending.pop(identity, None)
            self._terminal.add(identity)
            return
        if isinstance(event, ToolCompleted):
            pending = self._pending.pop(identity, None)
            self._terminal.add(identity)
            if pending is None or pending.tool_name != event.tool_name:
                return
            self._successful_paths.append(pending.path)

    @property
    def successful_paths(self) -> tuple[str, ...]:
        """Return successful paths in completion order."""

        return tuple(self._successful_paths)

    def _observe_call(
        self,
        identity: _CallIdentity,
        event: ToolCallCompleted,
    ) -> None:
        if event.tool_name not in _WRITE_TOOLS or identity in self._terminal:
            return
        path = event.arguments.get("path")
        if not isinstance(path, str) or not path:
            return
        candidate = _PendingWrite(tool_name=event.tool_name, path=path)
        existing = self._pending.get(identity)
        if existing is not None and existing != candidate:
            self._pending.pop(identity, None)
            self._terminal.add(identity)
            return
        self._pending[identity] = candidate

    @staticmethod
    def _identity(event: RuntimeEvent) -> _CallIdentity | None:
        if not isinstance(event, (ToolCallCompleted, ToolCompleted, ToolFailed)):
            return None
        return (event.session_id, event.run_id, event.iteration, event.tool_call_id)


def sha256_file(path: Path) -> str:
    """Hash a file without loading an unbounded artifact into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_primary_result(
    *,
    workspace: Workspace,
    evolution_id: str,
    version: str,
    conversation: ConversationState,
    collector: EpisodeArtifactCollector,
) -> ArtifactRef:
    """Select the registered ``result.md`` or create it from final assistant text.

    No directory scan participates in selection. A pre-existing result is used
    only when a correlated successful write/edit event names that exact resolved
    path; otherwise fallback creation uses exclusive-create semantics.
    """

    validate_managed_id(evolution_id)
    if not _EPISODE_VERSION.fullmatch(version):
        raise ValueError("episode version must match vNNN")
    relative = f"user_output/{evolution_id}/{version}/{_RESULT_NAME}"
    result_path = workspace.resolve(relative, must_exist=False)
    canonical_path = workspace.root / relative
    if result_path != canonical_path:
        raise EpisodeArtifactError(
            f"episode result path is not the canonical managed path: {relative}"
        )
    registered = _registered_exact_result(
        workspace=workspace,
        collector=collector,
        result_path=result_path,
    )
    if registered:
        if not result_path.is_file():
            raise MissingEpisodeResultError(
                f"registered episode result is missing or not a file: {relative}"
            )
        return _artifact_ref(workspace, result_path)

    final_text = _last_nonempty_assistant_text(conversation)
    if final_text is None:
        raise MissingEpisodeResultError(
            "episode produced neither a registered result.md nor nonempty assistant text"
        )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    payload = final_text if final_text.endswith("\n") else f"{final_text}\n"
    try:
        with result_path.open("x", encoding="utf-8", newline="") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise EpisodeResultAlreadyExistsError(
            f"episode result already exists and was not registered: {relative}"
        ) from exc
    return _artifact_ref(workspace, result_path)


def _registered_exact_result(
    *,
    workspace: Workspace,
    collector: EpisodeArtifactCollector,
    result_path: Path,
) -> bool:
    registered = False
    for raw_path in collector.successful_paths:
        try:
            candidate = workspace.resolve(raw_path, must_exist=False)
        except (OSError, ValueError, ToolExecutionError):
            continue
        if candidate == result_path:
            registered = True
    return registered


def _last_nonempty_assistant_text(conversation: ConversationState) -> str | None:
    for message in reversed(conversation.messages):
        if isinstance(message, AssistantMessage) and message.text.strip():
            return message.text
    return None


def _artifact_ref(workspace: Workspace, path: Path) -> ArtifactRef:
    stat = path.stat()
    return ArtifactRef(
        path=workspace.relative(path),
        media_type="text/markdown",
        size_bytes=stat.st_size,
        sha256=sha256_file(path),
    )


__all__ = [
    "EpisodeArtifactCollector",
    "EpisodeArtifactError",
    "EpisodeResultAlreadyExistsError",
    "MissingEpisodeResultError",
    "materialize_primary_result",
    "sha256_file",
]
