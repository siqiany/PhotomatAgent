from __future__ import annotations

import json

import pytest

from photomatagent.models.fake import FakeModelProvider, FakeResponse
from photomatagent.models.types import SystemMessage
from photomatagent.runtime.budget import BudgetState
from photomatagent.runtime.loop import AgentRuntime
from photomatagent.runtime.permissions import AllowAllPolicy
from photomatagent.scientific.capabilities.contracts import ScientificEvidence
from photomatagent.scientific.loop.candidate import candidate_from_formula
from photomatagent.scientific.loop.controller import (
    ScientificLoopConfig,
    ScientificLoopController,
)
from photomatagent.scientific.loop.evaluation import ScientificEvaluator
from photomatagent.scientific.loop.feedback import build_feedback
from photomatagent.scientific.loop.judge import (
    JudgeIssue,
    JudgeReport,
    ScientificJudge,
)
from photomatagent.scientific.loop.policy import (
    ScientificLoopPolicy,
    ScientificLoopState,
)
from photomatagent.scientific.loop.stagnation import StagnationDetector
from photomatagent.scientific.loop.target import (
    TargetSpec,
    canonical_lwir_detector_target,
)
from photomatagent.scientific.state import ScientificState
from photomatagent.tools.factory import create_default_registry
from photomatagent.workspace import Workspace


def _target() -> TargetSpec:
    return canonical_lwir_detector_target()


def _evaluated(band_gap: float | None, responsivity: float | None):
    candidate = candidate_from_formula("HgTe")
    state = ScientificState()
    if band_gap is not None:
        state.add_evidence(
            ScientificEvidence(
                subject="HgTe",
                property="band_gap",
                value=band_gap,
                unit="eV",
                source="s",
                source_type="dft_calculation",
                fidelity="dft",
            )
        )
    if responsivity is not None:
        state.add_evidence(
            ScientificEvidence(
                subject="HgTe",
                property="responsivity",
                value=responsivity,
                unit="A/W",
                source="s2",
                source_type="experimental",
                fidelity="experimental",
            )
        )
    return candidate, state, ScientificEvaluator(_target()).evaluate(candidate, state)


def _judge_report_json(
    quality: float = 0.9, issues: list[dict] | None = None
) -> str:
    return json.dumps(
        {
            "scientific_quality": quality,
            "issues": issues or [],
            "recommendations": ["validate the detector stack"],
            "rationale": "advisory assessment",
        }
    )


# ---------------------------------------------------------------------- #
# model + judge unit behavior
# ---------------------------------------------------------------------- #


def test_judge_report_model_validation():
    report = JudgeReport.model_validate(
        {
            "scientific_quality": 0.7,
            "issues": [
                {
                    "category": "realizability",
                    "severity": "HIGH",
                    "property": "responsivity",
                    "description": "dark current unvalidated",
                }
            ],
        }
    )
    assert report.available
    assert report.scientific_quality == 0.7
    assert len(report.significant_issues) == 1
    issue = report.issues[0]
    assert isinstance(issue, JudgeIssue)
    assert issue.category == "realizability"
    assert issue.severity == "HIGH"


def test_judge_report_low_issues_are_not_significant():
    report = JudgeReport(
        scientific_quality=0.8,
        issues=[
            JudgeIssue(category="other", severity="LOW", description="minor")
        ],
    )
    assert report.significant_issues == []


@pytest.mark.asyncio
async def test_judge_parses_scripted_json():
    candidate, state, evaluation = _evaluated(0.12, 1.4)
    model = FakeModelProvider([FakeResponse(text=_judge_report_json(0.85))])
    judge = ScientificJudge(model=model)
    report = await judge.assess(
        target=_target(),
        candidate=candidate,
        scientific=state,
        evaluation=evaluation,
        round_number=2,
    )
    assert report.status == "AVAILABLE"
    assert report.scientific_quality == 0.85
    assert report.candidate_id == candidate.candidate_id
    assert report.provider == "fake"
    assert report.model == "fake"


@pytest.mark.asyncio
async def test_judge_request_has_no_tools_and_uses_its_own_provider():
    candidate, state, evaluation = _evaluated(0.12, 1.4)
    model = FakeModelProvider([FakeResponse(text=_judge_report_json())])
    judge = ScientificJudge(model=model)
    await judge.assess(
        target=_target(),
        candidate=candidate,
        scientific=state,
        evaluation=evaluation,
        round_number=1,
    )
    assert len(model.requests) == 1
    request = model.requests[0]
    assert request.tools == []  # isolated: the judge can never call tools
    assert isinstance(request.messages[0], SystemMessage)


@pytest.mark.asyncio
async def test_judge_is_read_only():
    candidate, state, evaluation = _evaluated(0.12, 1.4)
    state_before = state.model_dump(mode="json")
    model = FakeModelProvider([FakeResponse(text=_judge_report_json(0.8))])
    judge = ScientificJudge(model=model)
    report = await judge.assess(
        target=_target(),
        candidate=candidate,
        scientific=state,
        evaluation=evaluation,
        round_number=1,
    )
    assert report.status == "AVAILABLE"
    assert state.model_dump(mode="json") == state_before  # nothing mutated


@pytest.mark.asyncio
async def test_judge_degrades_on_invalid_output():
    candidate, state, evaluation = _evaluated(0.12, 1.4)
    judge = ScientificJudge(
        model=FakeModelProvider([FakeResponse(text="I am not JSON at all")])
    )
    report = await judge.assess(
        target=_target(),
        candidate=candidate,
        scientific=state,
        evaluation=evaluation,
        round_number=1,
    )
    assert report.status == "UNAVAILABLE"
    assert not report.available
    assert report.error


@pytest.mark.asyncio
async def test_judge_degrades_on_provider_failure():
    class BrokenJudgeModel:
        provider = "broken"
        model = "broken"

        async def stream(self, request):
            if False:
                yield  # pragma: no cover
            raise RuntimeError("judge provider exploded")

    candidate, state, evaluation = _evaluated(0.12, 1.4)
    judge = ScientificJudge(model=BrokenJudgeModel())
    report = await judge.assess(
        target=_target(),
        candidate=candidate,
        scientific=state,
        evaluation=evaluation,
        round_number=1,
    )
    assert report.status == "UNAVAILABLE"
    assert "exploded" in report.error


# ---------------------------------------------------------------------- #
# feedback integration: judge embedded, never authoritative
# ---------------------------------------------------------------------- #


def test_feedback_embeds_judge_concerns_but_keeps_hard_violations():
    candidate, state, evaluation = _evaluated(0.21, None)  # deterministic FAIL
    judge = JudgeReport(
        candidate_id=candidate.candidate_id,
        scientific_quality=1.0,
        issues=[],  # glowing judge
        rationale="everything is fine",
    )
    signal = build_feedback(_target(), candidate, evaluation, [], judge=judge)
    assert signal is not None
    assert len(signal.violations) >= 1  # judge cannot rescind the violation
    assert signal.judge is not None
    # judge actions must never outrank deterministic work
    priority = [a.priority for a in signal.recommended_actions]
    assert priority == sorted(priority)


def test_feedback_judge_concerns_only_raise_concerns():
    candidate, state, evaluation = _evaluated(0.12, 1.4)  # deterministic PASS
    judge = JudgeReport(
        candidate_id=candidate.candidate_id,
        scientific_quality=0.2,
        issues=[
            JudgeIssue(
                category="realizability",
                severity="HIGH",
                property="responsivity",
                description="dark current unvalidated",
            )
        ],
    )
    signal = build_feedback(_target(), candidate, evaluation, [], judge=judge)
    assert signal is not None  # PASS + concerns -> actionable feedback
    assert signal.decision == "CONTINUE"
    assert any(a.action_type == "VALIDATE" for a in signal.recommended_actions)
    assert "Judge concerns" in signal.summary


def test_feedback_judge_without_concerns_on_pass_yields_none():
    candidate, _, evaluation = _evaluated(0.12, 1.4)
    judge = JudgeReport(candidate_id="c", scientific_quality=0.95)
    signal = build_feedback(_target(), candidate, evaluation, [], judge=judge)
    assert signal is None  # nothing actionable; policy decides SUCCESS


# ---------------------------------------------------------------------- #
# policy integration: judge can only hold SUCCESS back
# ---------------------------------------------------------------------- #


def _decide(evaluation, judge, *, require_judge=False, min_quality=0.6):
    return ScientificLoopPolicy(
        judge_min_quality=min_quality, require_judge=require_judge
    ).decide(
        evaluation=evaluation,
        state=ScientificLoopState(target=_target()),
        stagnation=StagnationDetector(),
        max_rounds=6,
        max_candidates=12,
        min_confidence=0.6,
        judge=judge,
    )


def test_policy_judge_cannot_rescue_deterministic_fail():
    _, _, evaluation = _evaluated(0.21, None)
    judge = JudgeReport(candidate_id="c", scientific_quality=1.0)
    decision = _decide(evaluation, judge)
    assert decision.action != "SUCCESS"


def test_policy_judge_quality_holds_back_success():
    _, _, evaluation = _evaluated(0.12, 1.4)  # deterministic PASS
    worried = JudgeReport(candidate_id="c", scientific_quality=0.3)
    assert _decide(evaluation, worried).action == "CONTINUE"
    ok = JudgeReport(candidate_id="c", scientific_quality=0.9)
    assert _decide(evaluation, ok).action == "SUCCESS"


def test_policy_unavailable_judge_does_not_block_success_by_default():
    _, _, evaluation = _evaluated(0.12, 1.4)
    unavailable = JudgeReport(status="UNAVAILABLE", error="solver down")
    assert _decide(evaluation, unavailable).action == "SUCCESS"
    assert (
        _decide(evaluation, unavailable, require_judge=True).action == "CONTINUE"
    )
    # A required judge that is not configured at all also blocks SUCCESS.
    assert (
        ScientificLoopPolicy(require_judge=True).decide(
            evaluation=evaluation,
            state=ScientificLoopState(target=_target()),
            stagnation=StagnationDetector(),
            max_rounds=6,
            max_candidates=12,
            min_confidence=0.6,
            judge=None,
        ).action
        == "CONTINUE"
    )


def test_policy_judge_gate_never_needs_evidence():
    # Even a judge with no evidence cannot manufacture a SUCCESS.
    _, _, evaluation = _evaluated(None, None)  # all UNKNOWN
    judge = JudgeReport(candidate_id="c", scientific_quality=1.0)
    assert _decide(evaluation, judge).action != "SUCCESS"


# ---------------------------------------------------------------------- #
# controller integration
# ---------------------------------------------------------------------- #


def _make_controller(
    maker_script,
    judge_script,
    *,
    max_rounds: int = 3,
    require_judge: bool = False,
):
    workspace = Workspace(".")
    scientific = ScientificState()
    registry = create_default_registry(scientific, workspace)
    runtime = AgentRuntime(
        model=FakeModelProvider(maker_script),
        tools=registry,
        workspace=workspace,
        scientific_state=scientific,
        permission_policy=AllowAllPolicy(),
        budget=BudgetState(max_iterations=25),
    )
    judge = ScientificJudge(model=FakeModelProvider(judge_script))
    return ScientificLoopController(
        target=_target(),
        runtime=runtime,
        config=ScientificLoopConfig(
            max_rounds=max_rounds,
            judge_min_quality=0.6,
            require_judge=require_judge,
        ),
        judge=judge,
        event_sinks=[],
    )


@pytest.mark.asyncio
async def test_controller_judge_runs_and_emits_candidate_judged():
    from photomatagent.models.types import ToolCall

    maker_script = [
        FakeResponse(
            tool_calls=[
                ToolCall(
                    name="tool_call",
                    arguments={
                        "name": "mock.run_calculation",
                        "arguments": {
                            "material": "InAs",
                            "calculation_type": "band_structure",
                        },
                    },
                )
            ]
        ),
        FakeResponse(text="proposal complete"),
        FakeResponse(text="keep working"),
        FakeResponse(text="final attempt"),
    ]
    judge_script = [
        FakeResponse(
            text=_judge_report_json(
                0.4,
                issues=[
                    {
                        "category": "realizability",
                        "severity": "HIGH",
                        "property": "responsivity",
                        "description": "detector stack not demonstrated",
                    }
                ],
            )
        )
    ]
    controller = _make_controller(maker_script, judge_script, max_rounds=3)
    events = []
    async for event in controller.run():
        events.append(event)
    kinds = [e.kind for e in events]
    assert "candidate_judged" in kinds
    judged = next(e for e in events if e.kind == "candidate_judged")
    assert judged.status == "AVAILABLE"
    assert judged.candidate_id
    assert controller.summary is not None
    assert controller.summary.judge_report is not None
    # mock band gap (0.31 eV empirical, confidence 0.5) fails hard -> never SUCCESS
    assert controller.summary.status != "SUCCESS"


@pytest.mark.asyncio
async def test_controller_require_judge_blocks_success_without_judge():
    """With require_judge, a deterministic pass must not succeed when the
    judge cannot run (missing candidate yearns for evidence)."""

    from photomatagent.models.types import ToolCall

    # maker proposes a candidate but no judge provider configured
    maker_script = [
        FakeResponse(
            tool_calls=[
                ToolCall(
                    name="tool_call",
                    arguments={
                        "name": "mock.run_calculation",
                        "arguments": {
                            "material": "InAs",
                            "calculation_type": "band_structure",
                        },
                    },
                )
            ]
        ),
        FakeResponse(text="proposal complete"),
    ]
    workspace = Workspace(".")
    scientific = ScientificState()
    registry = create_default_registry(scientific, workspace)
    runtime = AgentRuntime(
        model=FakeModelProvider(maker_script),
        tools=registry,
        workspace=workspace,
        scientific_state=scientific,
        permission_policy=AllowAllPolicy(),
        budget=BudgetState(max_iterations=25),
    )
    controller = ScientificLoopController(
        target=_target(),
        runtime=runtime,
        config=ScientificLoopConfig(max_rounds=1, require_judge=True),
        # judge is None: require_judge has no provider to judge with
        event_sinks=[],
    )
    async for _ in controller.run():
        pass
    # deterministic evaluation never PASSed anyway (mock 0.31 eV hard fail);
    # the important assertion: no crash, no fabricated SUCCESS.
    assert controller.summary is not None
    assert controller.summary.status != "SUCCESS"


def test_controller_judge_issue_flows_into_feedback_signal():
    candidate, _, evaluation = _evaluated(0.12, 1.4)
    signal = build_feedback(
        _target(),
        candidate,
        evaluation,
        [],
        judge=JudgeReport(
            scientific_quality=0.2,
            issues=[
                JudgeIssue(
                    category="methodology",
                    severity="MEDIUM",
                    description="band gap from a single phase",
                )
            ],
        ),
    )
    assert signal is not None
    text = signal.summary
    assert "Judge concerns" in text
    assert "band gap from a single phase" in text