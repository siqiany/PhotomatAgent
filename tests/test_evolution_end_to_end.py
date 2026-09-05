from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from photomatagent.models.fake import FakeModelProvider, FakeResponse, scripted_tool_call
from photomatagent.models.types import ModelRequest, UserMessage
from photomatagent.runtime.budget import BudgetState
from photomatagent.runtime.events import RuntimeEvent, parse_event
from photomatagent.runtime.loop import AgentRuntime
from photomatagent.runtime.permissions import (
    AllowAllPolicy,
    DenyAllPolicy,
    SwitchablePermissionPolicy,
)
from photomatagent.scientific.capabilities.contracts import (
    ScientificEvidence,
    ScientificToolResult,
)
from photomatagent.scientific.evolution import (
    FeedbackCompiler,
    ScientificEpisodeExecutor,
    build_inherited_scientific_state,
    build_revision_plan,
    run_fresh_evaluation,
)
from photomatagent.scientific.evolution.models import (
    EpisodeRecord,
    EvolutionTask,
    ExpertFeedbackDraft,
    FeedbackCompilation,
    RevisionPlan,
    RubricScores,
)
from photomatagent.scientific.evolution.service import EvolutionService, MutationResult
from photomatagent.scientific.evolution.store import EvolutionStore
from photomatagent.scientific.loop import ScientificLoopConfig, TargetSpec
from photomatagent.scientific.state import ScientificState
from photomatagent.tools.base import Tool
from photomatagent.tools.exposure import ToolExposure
from photomatagent.tools.factory import create_default_registry
from photomatagent.tools.registry import ToolRegistry
from photomatagent.workspace import Workspace


RAW_EXPERT_TEXT = "专家原始自由文本：第一版没有给出可追溯的高保真证据，请修订。"
SMOKE_PATH = (
    Path(__file__).parents[1]
    / "experiments"
    / "expert-feedback-evolution-smoke.json"
)


class OfflineEvidenceTool(Tool):
    """Deterministic test-only evidence source; it never touches network or HPC."""

    name = "test.offline_evidence"
    namespace = "test"
    description = "Return fixed structured material evidence for an offline E2E test."
    exposure = ToolExposure.DIRECT
    input_schema = {
        "type": "object",
        "properties": {
            "material": {"type": "string"},
            "band_gap": {"type": "number"},
            "fidelity": {"type": "string", "enum": ["dft", "experimental"]},
        },
        "required": ["material", "band_gap", "fidelity"],
    }

    async def execute(self, arguments: dict[str, Any]) -> ScientificToolResult:
        material = str(arguments["material"])
        band_gap = float(arguments["band_gap"])
        fidelity = str(arguments["fidelity"])
        suffix = str(round(band_gap * 1000))
        source_type = "experimental" if fidelity == "experimental" else "dft_calculation"
        evidence = [
            ScientificEvidence(
                id=f"sev_formula_{suffix}",
                subject=material,
                property="candidate_formula",
                value=material,
                source="offline test fixture",
                source_type=source_type,
                method="fixed deterministic fixture",
                fidelity=fidelity,
                provenance={"validated": True, "offline": True},
            ),
            ScientificEvidence(
                id=f"sev_band_gap_{suffix}",
                subject=material,
                property="band_gap",
                value=band_gap,
                unit="eV",
                source="offline test fixture",
                source_type=source_type,
                method="fixed deterministic fixture",
                fidelity=fidelity,
                provenance={"validated": True, "offline": True},
            ),
        ]
        return ScientificToolResult(output="offline evidence recorded", evidence=evidence)


def episode_v1_responses() -> list[FakeResponse]:
    return [
        scripted_tool_call(
            "test.offline_evidence",
            {"material": "GaAs", "band_gap": 0.31, "fidelity": "dft"},
        ),
        FakeResponse(text="v001 离线报告：候选仍违反带隙硬约束。"),
    ]


def episode_v2_responses() -> list[FakeResponse]:
    return [
        scripted_tool_call(
            "test.offline_evidence",
            {"material": "GaAs", "band_gap": 0.10, "fidelity": "experimental"},
        ),
        FakeResponse(text="v002 离线报告：新增高保真证据并关闭机器验收项。"),
    ]


def compiler_response() -> list[FakeResponse]:
    payload = {
        "status": "AVAILABLE",
        "items": [
            {
                "category": "EVIDENCE_SUFFICIENCY",
                "status": "CORRECTION",
                "severity": "HIGH",
                "responsible_module": "retrieval_planner",
                "problem": "第一版缺少高保真证据链",
                "requested_actions": ["补充结构化高保真证据"],
                "acceptance_test": "artifact_present",
                "preserve": ["保留已验证的任务目标"],
                "confidence": 1.0,
                "source_span": "第一版没有给出可追溯的高保真证据",
            }
        ],
        "warnings": [],
    }
    return [FakeResponse(text=json.dumps(payload, ensure_ascii=False))]


def expert_review() -> ExpertFeedbackDraft:
    return ExpertFeedbackDraft(
        scores=RubricScores(
            scientific_correctness=5,
            evidence_sufficiency=5,
            novelty=5,
            actionability=5,
            overall=5,
        ),
        comments=RAW_EXPERT_TEXT,
        priority_corrections=["补充高保真证据"],
        preserved_strengths=["保留目标定义"],
        recommended_actions=["运行离线验证"],
    )


def _smoke_spec() -> dict[str, Any]:
    return json.loads(SMOKE_PATH.read_text(encoding="utf-8"))


def _target_from_smoke() -> TargetSpec:
    return TargetSpec.model_validate(_smoke_spec()["input"]["target"])


def _serialize_requests(requests: list[ModelRequest]) -> str:
    return "\n".join(
        message.model_dump_json()
        for request in requests
        for message in request.messages
    )


def _assert_unique_ordered_event_subsequence(
    actual_payloads: list[dict[str, Any]],
    workflow_order: list[dict[str, Any]],
) -> None:
    unique_indices: list[int] = []
    for selector in workflow_order:
        matching_indices = [
            index
            for index, payload in enumerate(actual_payloads)
            if all(payload.get(key) == value for key, value in selector.items())
        ]
        assert len(matching_indices) == 1, (
            f"workflow selector {selector!r} matched indices {matching_indices!r}; "
            "expected exactly one event"
        )
        unique_indices.append(matching_indices[0])

    assert all(
        earlier < later
        for earlier, later in zip(unique_indices, unique_indices[1:], strict=False)
    ), (
        f"workflow selectors are out of order at indices {unique_indices!r}: "
        f"{workflow_order!r}"
    )


class EvolutionHarness:
    """A process-shaped adapter that only composes production workflow APIs."""

    def __init__(self, workspace: Path, responses: list[FakeResponse]) -> None:
        self.workspace = Workspace(workspace)
        self.store = EvolutionStore(self.workspace)
        self.events: list[RuntimeEvent] = []
        self.service = EvolutionService(self.store, event_sink=self.events.append)
        self.model = FakeModelProvider(responses)
        self.executor = ScientificEpisodeExecutor(self.store)
        self.last_runtime: AgentRuntime | None = None
        self.last_compilation: FeedbackCompilation | None = None
        self.last_revision: RevisionPlan | None = None
        self.feedback_entry_event_kinds: list[str] = []
        self.fresh_initial_state: ScientificState | None = None
        self.fresh_model: FakeModelProvider | None = None

    def _episode_runtime(
        self,
        *,
        session_id: str,
        scientific_state: ScientificState | None = None,
    ) -> AgentRuntime:
        state = scientific_state if scientific_state is not None else ScientificState()
        registry = ToolRegistry()
        registry.register(OfflineEvidenceTool())
        runtime = AgentRuntime(
            model=self.model,
            tools=registry,
            workspace=self.workspace,
            scientific_state=state,
            permission_policy=AllowAllPolicy(),
            budget=BudgetState(max_iterations=10),
            session_id=session_id,
        )
        self.last_runtime = runtime
        return runtime

    async def start(self, *, target: TargetSpec, goal: str) -> EvolutionTask:
        spec = _smoke_spec()
        created = self.service.create_task(
            goal=goal,
            target=target,
            evolution_id=spec["input"]["evolution_id"],
            task_group_id=spec["input"]["task_group_id"],
        )
        await self.service.publish(created)
        reserved = self.service.reserve_episode(
            created.entity.evolution_id,
            mode="NORMAL",
            provider="fake",
            model="fake",
        )
        await self.service.publish(reserved)
        await self.executor.execute(
            task=created.entity,
            episode=reserved.entity,
            runtime=self._episode_runtime(session_id="session_e2e_v001"),
            config=ScientificLoopConfig(max_rounds=1),
            on_event=self.events.append,
        )
        return self.service.get(created.entity.evolution_id)

    async def record_feedback(
        self,
        evolution_id: str,
        draft: ExpertFeedbackDraft,
    ) -> None:
        task = self.service.get(evolution_id)
        assert task.last_completed_version is not None
        episode = self.store.load_episode(evolution_id, task.last_completed_version)
        assert episode.artifact is not None
        requests_before = len(self.model.requests)
        journal_before = len(self.store.read_event_journal(evolution_id))
        recorded = self.service.attach_feedback(
            evolution_id,
            episode.version,
            feedback_id="fb_e2e_v001",
            draft=draft,
            result_sha256=episode.artifact.sha256,
            raw_input=RAW_EXPERT_TEXT,
        )
        await self.service.publish(recorded)
        assert len(self.model.requests) == requests_before
        new_records = self.store.read_event_journal(evolution_id)[journal_before:]
        self.feedback_entry_event_kinds = [
            str(envelope["event"]["kind"]) for _raw, envelope in new_records
        ]

    async def compile_feedback(self, evolution_id: str) -> FeedbackCompilation:
        task, episode, feedback = self.service.compilation_context(evolution_id)
        assert episode.artifact is not None
        result_text = self.workspace.resolve(
            episode.artifact.path, must_exist=True
        ).read_text(encoding="utf-8")
        compilation = await FeedbackCompiler(self.model).compile(
            task=task,
            episode=episode,
            feedback=feedback,
            result_text=result_text,
        )
        saved = self.service.save_compilation(evolution_id, compilation)
        await self.service.publish(saved)
        self.last_compilation = saved.entity
        return saved.entity

    async def record_and_compile_feedback(
        self,
        evolution_id: str,
        draft: ExpertFeedbackDraft,
    ) -> FeedbackCompilation:
        await self.record_feedback(evolution_id, draft)
        return await self.compile_feedback(evolution_id)

    async def confirm_revision(self, evolution_id: str) -> RevisionPlan:
        task, episode, feedback = self.service.compilation_context(evolution_id)
        compilation = self.service.available_compilation(
            evolution_id, feedback.feedback_id
        )
        assert compilation is not None
        plan = build_revision_plan(
            feedback=feedback,
            compilation=compilation,
            target=episode.target_snapshot,
            previous_summary=episode.summary,
        ).model_copy(update={"confirmed": True})
        confirmed = self.service.confirm_revision(evolution_id, plan)
        await self.service.publish(confirmed)
        self.last_revision = confirmed.entity
        return confirmed.entity

    async def iterate(self, evolution_id: str) -> EpisodeRecord:
        owner_token = "owner_e2e_v002"
        claim = self.service.claim_iteration(
            evolution_id,
            owner_token=owner_token,
            mode="CARRY_VERIFIED_EVIDENCE",
            provider="fake",
            model="fake",
        )
        await self.service.publish(MutationResult(claim.episode, claim.events))
        inherited, _decisions = build_inherited_scientific_state(
            claim.context.previous_scientific_state,
            source_episode=claim.context.source_episode.version,
            invalidated_evidence_ids=claim.context.revision.invalidated_evidence_ids,
            subject="GaAs",
        )
        result = await self.executor.execute(
            task=claim.context.task,
            episode=claim.episode,
            revision=claim.context.revision,
            runtime=self._episode_runtime(
                session_id="session_e2e_v002",
                scientific_state=inherited,
            ),
            config=ScientificLoopConfig(max_rounds=1),
            on_event=self.events.append,
            owner_token=owner_token,
        )
        return result.episode

    async def fresh_evaluate(self, evolution_id: str) -> EpisodeRecord:
        task = self.service.get(evolution_id)
        strategy_id = task.strategy_ids[-1]

        def runtime_factory(**kwargs: Any) -> AgentRuntime:
            state = kwargs["scientific_state"]
            assert isinstance(state, ScientificState)
            self.fresh_initial_state = state.model_copy(deep=True)
            isolated_workspace = Workspace(kwargs["workspace_root"])
            provider = self.model
            self.fresh_model = provider
            approval_root = isolated_workspace.resolve(
                str(kwargs["application_approval_root"]), must_exist=False
            )
            return AgentRuntime(
                model=provider,
                tools=create_default_registry(
                    state,
                    isolated_workspace,
                    evaluation_isolation=True,
                ),
                workspace=isolated_workspace,
                scientific_state=state,
                permission_policy=SwitchablePermissionPolicy(
                    DenyAllPolicy(), settings=None
                ),
                budget=BudgetState(max_iterations=10),
                session_id="session_e2e_fresh",
                fresh_approval=True,
                application_approval_root=approval_root,
            )

        result = await run_fresh_evaluation(
            service=self.service,
            task=task,
            strategy_id=strategy_id,
            executor=self.executor,
            runtime_factory=runtime_factory,
            config=ScientificLoopConfig(max_rounds=1),
            owner_token="owner_e2e_fresh",
            provider="fake",
            model="fake",
            on_event=self.events.append,
        )
        return result.episode


@pytest.mark.asyncio
async def test_async_expert_feedback_iteration_round_trip(tmp_path: Path) -> None:
    smoke = _smoke_spec()
    target = _target_from_smoke()

    first_process = EvolutionHarness(tmp_path, episode_v1_responses())
    task = await first_process.start(target=target, goal=smoke["input"]["goal"])
    assert task.status == "AWAITING_EXPERT_FEEDBACK"
    v1 = first_process.store.load_episode(task.evolution_id, "v001")
    assert v1.runtime_session_id is not None
    assert v1.summary is not None
    assert v1.summary.final_evaluation is not None
    assert v1.summary.final_evaluation.verdict == "FAIL"

    second_process = EvolutionHarness(tmp_path, compiler_response())
    compilation = await second_process.record_and_compile_feedback(
        task.evolution_id, expert_review()
    )
    assert compilation.status == "AVAILABLE"
    assert second_process.model.requests[0].tools == []
    revision = await second_process.confirm_revision(task.evolution_id)
    assert revision.confirmed is True
    ready_for_fresh = second_process.service.get(task.evolution_id)
    assert ready_for_fresh.status == smoke["expected"]["fresh_required_status"]
    assert ready_for_fresh.current_version == "v001"

    evaluation_process = EvolutionHarness(
        tmp_path, [FakeResponse(text="fresh evaluation without task history")]
    )
    fresh = await evaluation_process.fresh_evaluate(task.evolution_id)
    assert fresh.execution_mode == "FRESH_EVALUATION"
    assert fresh.applied_feedback_id is None
    assert fresh.revision_plan_id is None
    assert evaluation_process.fresh_initial_state == ScientificState()
    assert evaluation_process.fresh_model is not None
    assert RAW_EXPERT_TEXT not in _serialize_requests(
        evaluation_process.fresh_model.requests
    )
    after_fresh = evaluation_process.service.get(task.evolution_id)
    assert after_fresh.status == smoke["expected"]["fresh_required_status"]
    assert after_fresh.current_version == "v001"
    assert after_fresh.current_version != smoke["expected"]["fresh_precedes_version"]
    assert after_fresh.current_evaluation_version == "v001"

    third_process = EvolutionHarness(tmp_path, episode_v2_responses())
    v2 = await third_process.iterate(task.evolution_id)
    reloaded = third_process.store.load_task(task.evolution_id)
    assert reloaded.status == smoke["expected"]["final_status"]
    assert v2.version == smoke["expected"]["fresh_precedes_version"]
    assert v2.version == smoke["expected"]["versions"][-1]
    assert v2.runtime_session_id != v1.runtime_session_id
    assert v2.parent_version == "v001"
    assert v2.applied_feedback_id is not None
    assert v2.artifact is not None and v2.artifact.sha256
    assert third_process.last_runtime is not None
    assert RAW_EXPERT_TEXT not in _serialize_requests(third_process.model.requests)
    assert all(
        RAW_EXPERT_TEXT not in message.model_dump_json()
        for message in third_process.last_runtime.conversation_state.messages
    )

    comparison = third_process.service.compare(task.evolution_id, "v001", "v002")
    await third_process.service.publish(comparison)
    assert comparison.entity.closure_rate == smoke["expected"]["closure_rate"]
    assert comparison.entity.phase == "PRE_FEEDBACK"

    parsed = [
        parse_event(envelope["event"])
        for _raw, envelope in third_process.store.read_event_journal(task.evolution_id)
    ]
    workflow_order = smoke["expected"]["workflow_order"]
    assert all(isinstance(step, dict) for step in workflow_order)
    actual_payloads = [event.model_dump(mode="json") for event in parsed]
    _assert_unique_ordered_event_subsequence(actual_payloads, workflow_order)


@pytest.mark.asyncio
async def test_feedback_entry_emits_no_model_or_tool_request(tmp_path: Path) -> None:
    first_process = EvolutionHarness(tmp_path, episode_v1_responses())
    task = await first_process.start(
        target=_target_from_smoke(), goal=_smoke_spec()["input"]["goal"]
    )
    feedback_process = EvolutionHarness(tmp_path, compiler_response())

    await feedback_process.record_feedback(task.evolution_id, expert_review())

    assert feedback_process.model.requests == []
    assert feedback_process.feedback_entry_event_kinds == ["expert_feedback_recorded"]
    assert "model_request_started" not in feedback_process.feedback_entry_event_kinds
    assert "tool_requested" not in feedback_process.feedback_entry_event_kinds


@pytest.mark.asyncio
async def test_normal_expert_sentence_does_not_update_evolution_store(
    tmp_path: Path,
) -> None:
    harness = EvolutionHarness(tmp_path, [FakeResponse(text="普通聊天回复")])
    before = harness.store.list_tasks()
    runtime = harness._episode_runtime(session_id="session_ordinary_chat")

    _events = [event async for event in runtime.run("专家说证据不足")]

    assert before == []
    assert harness.store.list_tasks() == []
    user_messages = [
        message.content
        for message in runtime.conversation_state.messages
        if isinstance(message, UserMessage)
    ]
    assert user_messages[-1] == "专家说证据不足"


@pytest.mark.asyncio
async def test_five_star_feedback_cannot_override_deterministic_fail(
    tmp_path: Path,
) -> None:
    first_process = EvolutionHarness(tmp_path, episode_v1_responses())
    task = await first_process.start(
        target=_target_from_smoke(), goal=_smoke_spec()["input"]["goal"]
    )
    before = first_process.store.load_episode(task.evolution_id, "v001")
    assert before.summary is not None
    assert before.summary.final_evaluation is not None
    assert before.summary.final_evaluation.verdict == "FAIL"

    feedback_process = EvolutionHarness(tmp_path, compiler_response())
    await feedback_process.record_feedback(task.evolution_id, expert_review())

    after = feedback_process.store.load_episode(task.evolution_id, "v001")
    feedback = feedback_process.store.list_feedback(task.evolution_id)[0]
    assert feedback.scores.overall == 5
    assert after.summary is not None
    assert after.summary.final_evaluation is not None
    assert after.summary.final_evaluation.verdict == "FAIL"


def test_workflow_event_matcher_rejects_duplicate_selector_matches() -> None:
    selector = {
        "kind": "expert_feedback_recorded",
        "episode_version": "v001",
        "execution_mode": "NORMAL",
    }
    actual = [
        {"kind": "evolution_task_created"},
        selector.copy(),
        selector.copy(),
        {
            "kind": "expert_feedback_compiled",
            "episode_version": "v001",
            "execution_mode": "NORMAL",
        },
    ]
    expected = [
        {"kind": "evolution_task_created"},
        selector,
        {
            "kind": "expert_feedback_compiled",
            "episode_version": "v001",
            "execution_mode": "NORMAL",
        },
    ]

    with pytest.raises(
        AssertionError,
        match=r"expert_feedback_recorded.*matched indices.*\[1, 2\]",
    ):
        _assert_unique_ordered_event_subsequence(actual, expected)
