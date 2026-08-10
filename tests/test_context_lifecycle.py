from __future__ import annotations

import json

import pytest

from photomatagent.models.fake import FakeModelProvider, FakeResponse, scripted_tool_call
from photomatagent.models.types import (
    AssistantMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from photomatagent.runtime.context import ContextBuilder
from photomatagent.runtime.context_engine import (
    CompactionState,
    ContextEngine,
    ContextEngineConfig,
    RelevantResource,
    has_inflight_tool_transaction,
)
from photomatagent.runtime.ledger import derive_working_ledger, format_working_ledger
from photomatagent.runtime.loop import AgentRuntime
from photomatagent.runtime.permissions import AllowAllPolicy
from photomatagent.runtime.state import ConversationState
from photomatagent.runtime.sensitive import SensitivePathPolicy
from photomatagent.scientific.state import ScientificState
from photomatagent.tools.surface import ToolSurfaceStats
from photomatagent.tools.base import Tool, ToolResult
from photomatagent.tools.exposure import ToolExposure
from photomatagent.tools.registry import ToolRegistry
from photomatagent.workspace import Workspace

from conftest import collect, make_runtime


def _surface() -> ToolSurfaceStats:
    return ToolSurfaceStats(
        registered_tools=0,
        direct_tools=0,
        deferred_tools=0,
        hidden_tools=0,
        direct_schema_chars=0,
        deferred_schema_chars=0,
        bridge_schema_chars=0,
        manifest_chars=0,
        visible_schema_chars=0,
        estimated_direct_schema_tokens=0,
        estimated_deferred_schema_tokens=0,
        estimated_bridge_schema_tokens=0,
        estimated_manifest_tokens=0,
        estimated_visible_schema_tokens=0,
        estimated_avoided_tokens=0,
    )


def _engine(*, summarizer=None, protect_recent_turns: int = 1) -> ContextEngine:
    return ContextEngine(
        config=ContextEngineConfig(
            context_limit_tokens=4_000,
            prune_trigger_ratio=0.50,
            compact_trigger_ratio=0.90,
            target_ratio=0.40,
            protect_recent_turns=protect_recent_turns,
        ),
        summarizer=summarizer,
    )


async def _build(engine: ContextEngine, conversation: ConversationState, *, force=False):
    return await engine.build(
        conversation=conversation,
        scientific=ScientificState(),
        context_builder=ContextBuilder(),
        capability_manifest="",
        surface=_surface(),
        session_id="session-test",
        force_compaction=force,
    )


@pytest.mark.asyncio
async def test_pruning_changes_only_working_copy_and_preserves_tool_pair():
    call = ToolCall(id="old-call", name="grep", arguments={"pattern": "needle"})
    huge = "match\n" * 2_000
    conversation = ConversationState(
        messages=[
            UserMessage(content="old turn"),
            AssistantMessage(tool_calls=[call]),
            ToolResultMessage(
                tool_call_id=call.id, tool_name="grep", content=huge
            ),
            AssistantMessage(text="old conclusion"),
            UserMessage(content="current turn"),
        ]
    )
    durable_before = conversation.model_dump_json()

    result = await _build(_engine(), conversation)

    assert conversation.model_dump_json() == durable_before
    working_result = next(
        message
        for message in result.messages
        if isinstance(message, ToolResultMessage)
    )
    assert "Previous tool output omitted" in working_result.content
    assert "session://session-test/tool-call/old-call" in working_result.content
    assistant_index = next(
        index
        for index, message in enumerate(result.messages)
        if isinstance(message, AssistantMessage) and message.tool_calls
    )
    result_index = result.messages.index(working_result)
    assert result_index == assistant_index + 1
    assert result.pruned_tool_results == 1


@pytest.mark.asyncio
async def test_recent_tool_result_is_protected_from_pruning():
    old_call = ToolCall(id="old", name="grep", arguments={"pattern": "old"})
    recent_call = ToolCall(id="recent", name="read", arguments={"path": "x.py"})
    conversation = ConversationState(
        messages=[
            UserMessage(content="old"),
            AssistantMessage(tool_calls=[old_call]),
            ToolResultMessage(tool_call_id="old", tool_name="grep", content="x" * 9_000),
            UserMessage(content="current"),
            AssistantMessage(tool_calls=[recent_call]),
            ToolResultMessage(
                tool_call_id="recent", tool_name="read", content="y" * 9_000
            ),
        ]
    )
    # A larger limit leaves the protected recent result intact after reclaiming old output.
    engine = ContextEngine(
        config=ContextEngineConfig(
            context_limit_tokens=6_000,
            prune_trigger_ratio=0.5,
            compact_trigger_ratio=0.95,
            target_ratio=0.45,
            protect_recent_turns=1,
        )
    )
    result = await _build(engine, conversation)
    contents = {
        message.tool_call_id: message.content
        for message in result.messages
        if isinstance(message, ToolResultMessage)
    }
    assert "Previous tool output omitted" in contents["old"]
    assert contents["recent"] == "y" * 9_000


@pytest.mark.asyncio
async def test_explicitly_protected_tool_result_is_not_pruned():
    call = ToolCall(id="important", name="read", arguments={"path": "evidence.txt"})
    conversation = ConversationState(
        messages=[
            UserMessage(content="old evidence"),
            AssistantMessage(tool_calls=[call]),
            ToolResultMessage(
                tool_call_id=call.id,
                tool_name="read",
                content="critical evidence " * 800,
                protected=True,
            ),
            UserMessage(content="current"),
        ]
    )
    engine = ContextEngine(
        config=ContextEngineConfig(
            context_limit_tokens=20_000,
            prune_trigger_ratio=0.10,
            compact_trigger_ratio=0.95,
            target_ratio=0.05,
            protect_recent_turns=1,
        )
    )
    result = await _build(engine, conversation)
    important = next(
        message
        for message in result.messages
        if isinstance(message, ToolResultMessage)
    )
    assert important.content.startswith("critical evidence")
    assert result.pruned_tool_results == 0


class StubSummarizer:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    async def summarize(self, messages, previous):
        self.calls += 1
        if self.fail:
            raise RuntimeError("summary unavailable")
        return CompactionState(
            goal="finish implementation",
            progress=["inspected runtime"],
            key_findings=["tool pairs are atomic"],
            relevant_resources=[
                RelevantResource(reference="src/runtime.py", relevance="implementation")
            ],
            next_actions=["run tests"],
        )


@pytest.mark.asyncio
async def test_inflight_transaction_blocks_semantic_compaction():
    summarizer = StubSummarizer()
    call = ToolCall(id="pending", name="read", arguments={"path": "x"})
    conversation = ConversationState(
        messages=[
            UserMessage(content="old"),
            AssistantMessage(text="done"),
            UserMessage(content="current"),
            AssistantMessage(text="x" * 20_000, tool_calls=[call]),
        ]
    )
    assert has_inflight_tool_transaction(conversation.messages)
    engine = ContextEngine(
        config=ContextEngineConfig(
            context_limit_tokens=100_000,
            prune_trigger_ratio=0.7,
            compact_trigger_ratio=0.9,
            target_ratio=0.6,
            protect_recent_turns=1,
        ),
        summarizer=summarizer,
    )
    result = await _build(engine, conversation, force=True)
    assert summarizer.calls == 0
    assert result.inflight_tool_transaction is True
    assert not any(event.kind.startswith("context_compaction") for event in result.events)


@pytest.mark.asyncio
async def test_structured_compaction_replaces_only_old_prefix():
    summarizer = StubSummarizer()
    conversation = ConversationState(
        messages=[
            UserMessage(content="old goal"),
            AssistantMessage(text="old answer"),
            UserMessage(content="current goal"),
            AssistantMessage(text="current evidence"),
        ]
    )
    durable_before = conversation.model_dump_json()
    result = await _build(_engine(summarizer=summarizer), conversation, force=True)
    assert conversation.model_dump_json() == durable_before
    assert summarizer.calls == 1
    assert result.compaction_count == 1
    assert any(
        getattr(message, "kind", "") == "system"
        and "Compaction Summary" in message.content
        for message in result.messages
    )
    assert any(
        isinstance(message, UserMessage) and message.content == "current goal"
        for message in result.messages
    )
    assert not any(
        isinstance(message, UserMessage) and message.content == "old goal"
        for message in result.messages
    )


@pytest.mark.asyncio
async def test_compaction_failure_retains_working_history():
    summarizer = StubSummarizer(fail=True)
    conversation = ConversationState(
        messages=[
            UserMessage(content="old goal"),
            AssistantMessage(text="old answer"),
            UserMessage(content="current goal"),
        ]
    )
    result = await _build(_engine(summarizer=summarizer), conversation, force=True)
    assert any(
        isinstance(message, UserMessage) and message.content == "old goal"
        for message in result.messages
    )
    assert result.compaction_count == 0
    assert any(event.kind == "context_compaction_failed" for event in result.events)


@pytest.mark.asyncio
async def test_sensitive_env_read_is_blocked_before_execution(tmp_path):
    (tmp_path / ".env").write_text("OPENAI_API_KEY=do-not-read\n", encoding="utf-8")
    model = FakeModelProvider(
        [
            scripted_tool_call("read", {"path": ".env"}, tool_call_id="secret-read"),
            FakeResponse(text="blocked"),
        ]
    )
    runtime = make_runtime(model, workspace=Workspace(tmp_path))
    events = await collect(runtime, "inspect provider config")
    assert any(event.kind == "sensitive_access_blocked" for event in events)
    assert not any(event.kind == "tool_started" for event in events)
    result = next(
        message
        for message in model.requests[1].messages
        if isinstance(message, ToolResultMessage)
    )
    assert result.is_error
    assert "do-not-read" not in json.dumps(model.requests[1].model_dump())


@pytest.mark.asyncio
async def test_bash_secret_stdout_is_redacted_before_model_context(tmp_path):
    class SecretOutputBash(Tool):
        name = "bash"
        description = "deterministic bash-shaped test tool"
        exposure = ToolExposure.DIRECT
        input_schema = {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        }

        async def execute(self, arguments):
            return ToolResult(
                output=(
                    "stdout:\nSERVICE_API_KEY=super-secret-value\n"
                    "Authorization: Bearer token-123\n"
                )
            )

    model = FakeModelProvider(
        [
            scripted_tool_call(
                "bash",
                {"command": "printf 'SERVICE_API_KEY=super-secret-value\\nAuthorization: Bearer token-123\\n'"},
                tool_call_id="bash-secret",
            ),
            FakeResponse(text="safe"),
        ]
    )
    registry = ToolRegistry()
    registry.register(SecretOutputBash())
    runtime = AgentRuntime(
        model=model,
        tools=registry,
        workspace=Workspace(tmp_path),
        permission_policy=AllowAllPolicy(),
    )
    events = await collect(runtime, "run diagnostic")
    completed = next(event for event in events if event.kind == "tool_completed")
    assert completed.redacted is True
    assert "super-secret-value" not in completed.output
    assert "token-123" not in completed.output
    delivered_result = next(
        message
        for message in model.requests[1].messages
        if isinstance(message, ToolResultMessage)
    )
    assert "super-secret-value" not in delivered_result.content
    assert "token-123" not in delivered_result.content


def test_working_ledger_is_derived_deduplicated_and_bounded():
    messages = []
    for index in range(20):
        call = ToolCall(
            id=f"call-{index}",
            name="grep",
            arguments={"pattern": f"needle-{index}", "path": "src"},
        )
        messages.extend(
            [
                AssistantMessage(tool_calls=[call]),
                ToolResultMessage(
                    tool_call_id=call.id,
                    tool_name="grep",
                    content=f"src/file.py:{index}:match",
                ),
            ]
        )
    ledger = derive_working_ledger(messages, max_chars=500, max_items=20)
    rendered = format_working_ledger(ledger)
    assert len(rendered) <= 500
    assert ledger.searched_queries
    assert ledger.inspected_paths == ["src"]
    assert ledger.key_observations


def test_sensitive_policy_blocks_credentials_without_treating_grep_query_as_path():
    policy = SensitivePathPolicy()
    for path in (
        ".env",
        ".env.local",
        "tls/client.pem",
        "tls/client.key",
        ".git-credentials",
        ".netrc",
        ".ssh/id_ed25519",
        ".aws/credentials",
        "credentials-prod.json",
        "secrets.yaml",
    ):
        assert policy.is_sensitive(path), path
    policy.check_tool_call("grep", {"pattern": "credentials", "path": "src"})
