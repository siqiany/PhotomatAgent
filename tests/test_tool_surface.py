from __future__ import annotations

import json

import pytest

from photomatagent.models.fake import FakeModelProvider, FakeResponse, scripted_tool_call
from photomatagent.models.types import ToolResultMessage
from photomatagent.runtime.context import ContextBuilder
from photomatagent.runtime.observation import ObservationPolicy, ObservationPolicyConfig
from photomatagent.runtime.permissions import PermissionDecision, PolicyRule
from photomatagent.runtime.state import ConversationState
from photomatagent.scientific.state import ScientificState
from photomatagent.skills.loader import SkillLoader
from photomatagent.tools.base import Tool, ToolResult
from photomatagent.tools.bridges import ToolDescribeTool
from photomatagent.tools.exposure import ToolExposure
from photomatagent.tools.factory import create_default_registry
from photomatagent.tools.registry import ToolRegistry
from photomatagent.tools.surface import (
    ToolCatalog,
    ToolSurfaceConfig,
    ToolSurfacePlanner,
    build_manifest,
)
from photomatagent.workspace import Workspace

from conftest import collect, make_runtime


class FakeCatalogTool(Tool):
    def __init__(
        self,
        name: str,
        description: str,
        *,
        exposure: ToolExposure = ToolExposure.DEFERRED,
        namespace: str = "test",
        tags: tuple[str, ...] = (),
        required: tuple[str, ...] = ("query",),
    ) -> None:
        self.name = name
        self.description = description
        self.short_description = description
        self.exposure = exposure
        self.namespace = namespace
        self.tags = tags
        self.input_schema = {
            "type": "object",
            "properties": {
                parameter: {"type": "string", "description": f"Value for {parameter}"}
                for parameter in required
            },
            "required": list(required),
        }

    async def execute(self, arguments: dict) -> ToolResult:
        return ToolResult(output=json.dumps(arguments, sort_keys=True))


class ExplodingTool(FakeCatalogTool):
    async def execute(self, arguments: dict) -> ToolResult:
        raise RuntimeError("x" * 2000)


def test_exposure_controls_model_visible_surface(tmp_path):
    registry = ToolRegistry()
    registry.register(FakeCatalogTool("core.read", "read files", exposure=ToolExposure.DIRECT))
    registry.register(FakeCatalogTool("literature.search", "search papers"))
    registry.register(FakeCatalogTool("disabled.tool", "disabled", exposure=ToolExposure.HIDDEN))

    surface = ToolSurfacePlanner(registry).plan()

    assert [tool.name for tool in surface.definitions] == ["core.read"]
    assert "literature.search" in surface.manifest.text
    assert "disabled.tool" not in surface.manifest.text
    assert surface.stats.registered_tools == 3
    assert surface.stats.direct_tools == 1
    assert surface.stats.deferred_tools == 1
    assert surface.stats.hidden_tools == 1


def test_bm25_search_over_twenty_fake_tools():
    registry = ToolRegistry()
    for index in range(20):
        registry.register(
            FakeCatalogTool(
                f"utility.tool_{index}",
                f"generic utility number {index}",
                tags=("utility",),
            )
        )
    registry.register(
        FakeCatalogTool(
            "literature.search",
            "Search scientific papers and scholarly literature.",
            namespace="literature",
            tags=("papers", "scientific", "search"),
        )
    )
    registry.register(
        FakeCatalogTool(
            "hpc.submit_calculation",
            "Submit a materials calculation to an HPC scheduler.",
            namespace="hpc",
            tags=("submit", "calculation", "cluster"),
        )
    )
    catalog = ToolCatalog(registry)

    papers = catalog.search("search scientific papers")
    submit = catalog.search("submit calculation")

    assert papers[0].entry.name == "literature.search"
    assert submit[0].entry.name == "hpc.submit_calculation"


@pytest.mark.asyncio
async def test_describe_returns_deferred_schema_but_not_direct_or_hidden():
    registry = ToolRegistry()
    registry.register(FakeCatalogTool("literature.search", "search papers"))
    registry.register(FakeCatalogTool("read", "read", exposure=ToolExposure.DIRECT))
    registry.register(FakeCatalogTool("secret", "secret", exposure=ToolExposure.HIDDEN))
    tool = ToolDescribeTool(ToolCatalog(registry))

    described = await tool.execute({"name": "literature.search"})
    direct = await tool.execute({"name": "read"})
    hidden = await tool.execute({"name": "secret"})

    assert not described.is_error
    assert described.data["required_parameters"] == ["query"]
    assert "schema" in described.data
    assert direct.is_error
    assert hidden.is_error


def test_manifest_never_exceeds_budget_and_degrades():
    registry = ToolRegistry()
    for index in range(120):
        registry.register(
            FakeCatalogTool(
                f"namespace_{index % 8}.very_long_tool_name_{index}",
                "A deliberately long capability description for budget testing. " * 4,
                namespace=f"namespace_{index % 8}",
            )
        )
    entries = ToolCatalog(registry).entries()

    manifest = build_manifest(entries, max_tokens=80)

    assert manifest.chars <= 320
    assert manifest.format in {"namespaces", "truncated"}
    assert "tool_search" in manifest.text


@pytest.mark.asyncio
async def test_bridge_executes_underlying_with_protocol_pairing(tmp_path):
    model = FakeModelProvider(
        [
            scripted_tool_call(
                "tool_call",
                {
                    "name": "mock.run_calculation",
                    "arguments": {
                        "material": "GaAs",
                        "calculation_type": "band_structure",
                    },
                },
                tool_call_id="bridge-call-1",
            ),
            FakeResponse(text="done"),
        ]
    )
    runtime = make_runtime(model, workspace=Workspace(tmp_path))

    events = await collect(runtime, "calculate")

    requested = next(event for event in events if event.kind == "tool_requested")
    completed = next(event for event in events if event.kind == "tool_completed")
    assert requested.tool_name == "mock.run_calculation"
    assert requested.bridge_tool == "tool_call"
    assert completed.underlying_tool == "mock.run_calculation"
    result = next(
        message
        for message in model.requests[1].messages
        if isinstance(message, ToolResultMessage)
    )
    assert result.tool_call_id == "bridge-call-1"
    assert result.tool_name == "tool_call"


@pytest.mark.asyncio
async def test_deferred_dangerous_tool_cannot_bypass_permission(tmp_path):
    model = FakeModelProvider(
        [
            scripted_tool_call(
                "tool_call",
                {"name": "mock.run_calculation", "arguments": {"material": "GaAs", "calculation_type": "dos"}},
            ),
            FakeResponse(text="denied"),
        ]
    )
    runtime = make_runtime(
        model,
        workspace=Workspace(tmp_path),
        permission_policy=PolicyRule(
            {"mock.run_calculation": PermissionDecision.DENY},
            default=PermissionDecision.ALLOW,
        ),
    )

    events = await collect(runtime, "calculate")

    denied = next(event for event in events if event.kind == "tool_permission_denied")
    assert denied.tool_name == "mock.run_calculation"
    assert denied.bridge_tool == "tool_call"
    assert not any(event.kind == "tool_started" for event in events)


@pytest.mark.asyncio
async def test_blind_deferred_call_returns_missing_args_and_parameter_help(tmp_path):
    model = FakeModelProvider(
        [
            scripted_tool_call(
                "tool_call",
                {"name": "mock.run_calculation", "arguments": {"material": "GaAs"}},
            ),
            FakeResponse(text="repaired"),
        ]
    )
    runtime = make_runtime(model, workspace=Workspace(tmp_path))

    await collect(runtime, "calculate")

    result = next(
        message
        for message in model.requests[1].messages
        if isinstance(message, ToolResultMessage)
    )
    payload = json.loads(result.content)
    assert payload["error"] == "missing_required_arguments"
    assert payload["missing"] == ["calculation_type"]
    assert "calculation_type" in payload["parameter_help"]


def test_skill_index_is_initially_compact_and_skill_view_loads_reference(tmp_path):
    skill_dir = tmp_path / "analysis"
    references = skill_dir / "references"
    references.mkdir(parents=True)
    secret_body = "FULL PROCEDURE SHOULD BE DEFERRED"
    (skill_dir / "SKILL.md").write_text(
        "---\nname: analysis\ndescription: Compact analysis guidance.\n"
        "category: science\ntags: [bands, materials]\n---\n" + secret_body,
        encoding="utf-8",
    )
    (references / "detail.md").write_text("REFERENCE DETAIL", encoding="utf-8")
    loader = SkillLoader(tmp_path)
    context = ContextBuilder(loader).build(
        conversation=ConversationState(),
        scientific=ScientificState(),
    )

    assert "Compact analysis guidance" in context[0].content
    assert secret_body not in context[0].content


@pytest.mark.asyncio
async def test_skill_view_loads_primary_and_one_reference(tmp_path):
    skill_dir = tmp_path / "analysis"
    references = skill_dir / "references"
    references.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: analysis\ndescription: Compact.\n---\nFULL PROCEDURE",
        encoding="utf-8",
    )
    (references / "detail.md").write_text("REFERENCE DETAIL", encoding="utf-8")
    loader = SkillLoader(tmp_path)
    registry = create_default_registry(
        ScientificState(), Workspace(tmp_path), skill_loader=loader
    )

    primary = await registry.get("skill_view").execute({"name": "analysis"})
    reference = await registry.get("skill_view").execute(
        {"name": "analysis", "path": "references/detail.md"}
    )

    assert "FULL PROCEDURE" in primary.output
    assert reference.output == "REFERENCE DETAIL"


def test_observation_policy_marks_truncation_and_bash_keeps_tail():
    policy = ObservationPolicy(
        ObservationPolicyConfig(
            default_max_chars=256,
            read_max_chars=256,
            grep_max_chars=256,
            glob_max_chars=256,
            bash_max_chars=256,
        )
    )
    output = "HEAD" + "x" * 1000 + "TAIL"

    observation = policy.apply("bash", output)

    assert observation.truncated
    assert observation.original_chars == len(output)
    assert observation.delivered_chars <= 256
    assert "output truncated" in observation.content
    assert observation.content.endswith("TAIL")


def test_observation_redacts_dotenv_before_model_visibility(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-model-visible-secret")
    policy = ObservationPolicy()
    raw = (
        "PHOTOMATAGENT_PROVIDER=openai\n"
        "OPENAI_MODEL=test\n"
        "OPENAI_API_KEY=sk-test-model-visible-secret\n"
    )

    observation = policy.apply("bash", raw)

    assert observation.redacted
    assert observation.content == "[REDACTED .env content]"
    assert "sk-test-model-visible-secret" not in observation.content


def test_progressive_surface_is_stable_across_plans(tmp_path):
    registry = create_default_registry(ScientificState(), Workspace(tmp_path))
    planner = ToolSurfacePlanner(registry, ToolSurfaceConfig())

    first = planner.plan()
    second = planner.plan()

    assert first.definitions == second.definitions
    assert first.stats == second.stats


def test_eager_surface_keeps_skill_view_but_removes_capability_bridges(tmp_path):
    registry = create_default_registry(ScientificState(), Workspace(tmp_path))
    names = {
        item.name
        for item in ToolSurfacePlanner(
            registry, ToolSurfaceConfig(mode="eager")
        ).plan().definitions
    }

    assert "skill_view" in names
    assert not {"tool_search", "tool_describe", "tool_call"} & names


@pytest.mark.asyncio
async def test_direct_deferred_call_is_rejected_and_not_executed(tmp_path):
    model = FakeModelProvider(
        [
            scripted_tool_call(
                "mock.run_calculation",
                {"material": "GaAs", "calculation_type": "band_structure"},
            ),
            FakeResponse(text="repaired"),
        ]
    )
    runtime = make_runtime(model, workspace=Workspace(tmp_path))

    events = await collect(runtime, "calculate")

    failed = next(event for event in events if event.kind == "tool_failed")
    assert failed.error_type == "DeferredToolDirectCall"
    assert "deferred_tool_requires_bridge" in failed.error
    assert not any(event.kind == "tool_started" for event in events)


@pytest.mark.asyncio
async def test_exception_text_is_observation_bounded(tmp_path):
    from photomatagent.runtime.loop import AgentRuntime
    from photomatagent.runtime.permissions import AllowAllPolicy

    registry = create_default_registry(ScientificState(), Workspace(tmp_path))
    registry.register(
        ExplodingTool("explode", "Raise a long error", exposure=ToolExposure.DIRECT)
    )
    model = FakeModelProvider(
        [scripted_tool_call("explode", {"query": "x"}), FakeResponse(text="done")]
    )
    runtime = AgentRuntime(
        model=model,
        tools=registry,
        workspace=Workspace(tmp_path),
        permission_policy=AllowAllPolicy(),
        observation_policy=ObservationPolicy(
            ObservationPolicyConfig(
                default_max_chars=256,
                read_max_chars=256,
                grep_max_chars=256,
                glob_max_chars=256,
                bash_max_chars=256,
            )
        ),
    )

    events = await collect(runtime, "explode")

    failed = next(event for event in events if event.kind == "tool_failed")
    assert failed.truncated
    assert failed.delivered_chars is not None and failed.delivered_chars <= 256
