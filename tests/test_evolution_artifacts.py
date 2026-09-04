from __future__ import annotations

from pathlib import Path

import pytest

from photomatagent.models.types import AssistantMessage, ModelMessage, UserMessage
from photomatagent.runtime.events import ToolCallCompleted, ToolCompleted, ToolFailed
from photomatagent.runtime.state import ConversationState
from photomatagent.scientific.evolution.artifacts import (
    EpisodeArtifactCollector,
    EpisodeArtifactError,
    EpisodeResultAlreadyExistsError,
    MissingEpisodeResultError,
    materialize_primary_result,
    sha256_file,
)
from photomatagent.workspace import Workspace


def _call(
    call_id: str,
    path: object,
    *,
    tool_name: str = "write",
    iteration: int = 1,
) -> ToolCallCompleted:
    return ToolCallCompleted(
        iteration=iteration,
        tool_call_id=call_id,
        tool_name=tool_name,
        arguments={"path": path, "content": "report"},
        index=0,
    )


def _completed(
    call_id: str,
    *,
    tool_name: str = "write",
    iteration: int = 1,
) -> ToolCompleted:
    return ToolCompleted(
        iteration=iteration,
        tool_call_id=call_id,
        tool_name=tool_name,
        output="created result",
    )


def _conversation(*texts: str) -> ConversationState:
    messages: list[ModelMessage] = [UserMessage(content="goal")]
    messages.extend(AssistantMessage(text=text) for text in texts)
    return ConversationState(messages=messages)


def test_fallback_result_is_last_nonempty_assistant_text(tmp_path: Path) -> None:
    conversation = _conversation("first", "  ", "final report")

    artifact = materialize_primary_result(
        workspace=Workspace(tmp_path),
        evolution_id="evo_test",
        version="v001",
        conversation=conversation,
        collector=EpisodeArtifactCollector(),
    )

    path = Workspace(tmp_path).resolve(artifact.path)
    assert path.read_text(encoding="utf-8") == "final report\n"
    assert artifact.path == "user_output/evo_test/v001/result.md"
    assert artifact.media_type == "text/markdown"
    assert artifact.size_bytes == 13
    assert artifact.sha256 == sha256_file(path)


def test_registered_result_requires_matching_successful_tool_call(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    path = workspace.resolve(
        "user_output/evo_test/v001/result.md", must_exist=False
    )
    path.parent.mkdir(parents=True)
    path.write_text("registered report", encoding="utf-8")
    collector = EpisodeArtifactCollector()
    collector.observe(
        _call(
            "call_result",
            "user_output/evo_test/v001/result.md",
            tool_name="edit",
        )
    )
    collector.observe(_completed("another_call"))

    with pytest.raises(EpisodeResultAlreadyExistsError):
        materialize_primary_result(
            workspace=workspace,
            evolution_id="evo_test",
            version="v001",
            conversation=_conversation("fallback must not overwrite"),
            collector=collector,
        )


def test_registered_result_requires_success_and_matching_tool_identity(
    tmp_path: Path,
) -> None:
    workspace = Workspace(tmp_path)
    path = workspace.resolve(
        "user_output/evo_test/v001/result.md", must_exist=False
    )
    path.parent.mkdir(parents=True)
    path.write_text("untrusted report", encoding="utf-8")
    collector = EpisodeArtifactCollector()
    collector.observe(_call("call_result", "user_output/evo_test/v001/result.md"))
    collector.observe(
        ToolFailed(
            iteration=1,
            tool_call_id="call_result",
            tool_name="write",
            error="failed",
        )
    )
    collector.observe(_completed("call_result", tool_name="edit"))

    with pytest.raises(EpisodeResultAlreadyExistsError):
        materialize_primary_result(
            workspace=workspace,
            evolution_id="evo_test",
            version="v001",
            conversation=_conversation("fallback"),
            collector=collector,
        )


def test_successful_registered_result_is_selected_without_rewriting(
    tmp_path: Path,
) -> None:
    workspace = Workspace(tmp_path)
    path = workspace.resolve(
        "user_output/evo_test/v001/result.md", must_exist=False
    )
    path.parent.mkdir(parents=True)
    path.write_text("registered report", encoding="utf-8")
    collector = EpisodeArtifactCollector()
    collector.observe(_call("call_result", "user_output/evo_test/v001/result.md"))
    collector.observe(_completed("call_result"))

    artifact = materialize_primary_result(
        workspace=workspace,
        evolution_id="evo_test",
        version="v001",
        conversation=_conversation("later assistant text"),
        collector=collector,
    )

    assert path.read_text(encoding="utf-8") == "registered report"
    assert artifact.path == "user_output/evo_test/v001/result.md"
    assert artifact.size_bytes == 17


@pytest.mark.parametrize(
    "event_path",
    [
        "user_output/evo_other/v001/result.md",
        "user_output/evo_test/v002/result.md",
        "user_output/evo_test/v001/other.md",
        "tmp/result.md",
        "user_output/evo_test/v001/../../outside.md",
    ],
)
def test_out_of_scope_write_event_cannot_select_primary_result(
    tmp_path: Path,
    event_path: str,
) -> None:
    workspace = Workspace(tmp_path)
    collector = EpisodeArtifactCollector()
    collector.observe(_call("call_outside", event_path))
    collector.observe(_completed("call_outside"))

    artifact = materialize_primary_result(
        workspace=workspace,
        evolution_id="evo_test",
        version="v001",
        conversation=_conversation("safe fallback"),
        collector=collector,
    )

    result_path = workspace.resolve(artifact.path)
    assert result_path.read_text(encoding="utf-8") == "safe fallback\n"


def test_absolute_outside_write_event_cannot_select_primary_result(
    tmp_path: Path,
) -> None:
    workspace = Workspace(tmp_path)
    collector = EpisodeArtifactCollector()
    collector.observe(_call("call_outside", str(tmp_path.parent / "result.md")))
    collector.observe(_completed("call_outside"))

    artifact = materialize_primary_result(
        workspace=workspace,
        evolution_id="evo_test",
        version="v001",
        conversation=_conversation("safe fallback"),
        collector=collector,
    )

    assert workspace.resolve(artifact.path).read_text(encoding="utf-8") == (
        "safe fallback\n"
    )


def test_symlink_aliased_result_is_not_the_canonical_episode_path(
    tmp_path: Path,
) -> None:
    workspace = Workspace(tmp_path)
    target = workspace.resolve("user_output/aliased.md", must_exist=False)
    target.write_text("aliased", encoding="utf-8")
    result = workspace.root / "user_output/evo_test/v001/result.md"
    result.parent.mkdir(parents=True)
    result.symlink_to(target)
    collector = EpisodeArtifactCollector()
    collector.observe(
        _call(
            "call_result",
            "user_output/evo_test/v001/result.md",
            tool_name="edit",
        )
    )
    collector.observe(_completed("call_result", tool_name="edit"))

    with pytest.raises(EpisodeArtifactError, match="canonical"):
        materialize_primary_result(
            workspace=workspace,
            evolution_id="evo_test",
            version="v001",
            conversation=_conversation("fallback"),
            collector=collector,
        )


def test_existing_unregistered_result_is_never_overwritten(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    path = workspace.resolve(
        "user_output/evo_test/v001/result.md", must_exist=False
    )
    path.parent.mkdir(parents=True)
    path.write_text("existing", encoding="utf-8")

    with pytest.raises(EpisodeResultAlreadyExistsError):
        materialize_primary_result(
            workspace=workspace,
            evolution_id="evo_test",
            version="v001",
            conversation=_conversation("replacement"),
            collector=EpisodeArtifactCollector(),
        )

    assert path.read_text(encoding="utf-8") == "existing"


def test_missing_registered_file_does_not_fall_back_or_scan(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    decoy = workspace.resolve(
        "user_output/evo_test/v001/decoy.md", must_exist=False
    )
    decoy.parent.mkdir(parents=True)
    decoy.write_text("decoy", encoding="utf-8")
    collector = EpisodeArtifactCollector()
    collector.observe(_call("call_result", "user_output/evo_test/v001/result.md"))
    collector.observe(_completed("call_result"))

    with pytest.raises(MissingEpisodeResultError):
        materialize_primary_result(
            workspace=workspace,
            evolution_id="evo_test",
            version="v001",
            conversation=_conversation("do not hide the lost registered result"),
            collector=collector,
        )


def test_no_registered_file_or_nonempty_assistant_text_is_missing(
    tmp_path: Path,
) -> None:
    with pytest.raises(MissingEpisodeResultError):
        materialize_primary_result(
            workspace=Workspace(tmp_path),
            evolution_id="evo_test",
            version="v001",
            conversation=_conversation("", " \n "),
            collector=EpisodeArtifactCollector(),
        )
