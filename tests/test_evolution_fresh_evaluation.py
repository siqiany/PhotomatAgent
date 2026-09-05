from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Barrier

import pytest
from pydantic import ValidationError

from photomatagent.models.fake import FakeModelProvider, FakeResponse
from photomatagent.runtime.budget import BudgetState
from photomatagent.runtime.loop import AgentRuntime
from photomatagent.runtime.permissions import DenyAllPolicy
from photomatagent.scientific.evolution.executor import (
    ScientificEpisodeExecutor,
    run_fresh_evaluation,
)
from photomatagent.scientific.evolution.models import (
    ArtifactRef,
    EpisodeRecord,
    ExpertFeedbackDraft,
    RubricScores,
    StrategyVersion,
)
from photomatagent.scientific.evolution.service import (
    EvolutionOperationConflict,
    EvolutionService,
    InvalidEvolutionTransition,
)
from photomatagent.scientific.evolution.store import (
    EvolutionAlreadyExistsError,
    EvolutionStore,
)
from photomatagent.scientific.loop import ScientificLoopConfig, ScientificLoopSummary
from photomatagent.scientific.loop.target import TargetSpec
from photomatagent.scientific.state import ScientificState
from photomatagent.tools.registry import ToolRegistry
from photomatagent.workspace import Workspace


def _complete_initial(service: EvolutionService, evolution_id: str) -> EpisodeRecord:
    task = service.create_task(
        goal="Produce a fresh-evaluation report",
        target=TargetSpec(goal="Produce a fresh-evaluation report"),
        evolution_id=evolution_id,
    ).entity
    reserved = service.reserve_episode(task.evolution_id, mode="NORMAL").entity
    running = service.mark_episode_running(
        task.evolution_id,
        reserved.version,
        runtime_session_id=f"session_{evolution_id}_initial",
    ).entity
    content = b"initial result\n"
    relative = f"user_output/{evolution_id}/v001/result.md"
    path = service.store.workspace.resolve(relative, must_exist=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    state_path = service.store.write_scientific_state(
        evolution_id,
        "v001",
        ScientificState(goal=task.goal, hypotheses=["task-specific history"]),
    )
    return service.complete_episode(
        task.evolution_id,
        running.version,
        result=running.model_copy(
            update={
                "scientific_state_path": service.store.workspace.relative(state_path),
                "summary": ScientificLoopSummary(
                    status="INCONCLUSIVE",
                    rounds=1,
                    candidate_count=0,
                    best_candidate_id=None,
                    best_score=0.0,
                    final_evaluation=None,
                ),
                "artifact": ArtifactRef(
                    path=relative,
                    size_bytes=len(content),
                    sha256=hashlib.sha256(content).hexdigest(),
                ),
            }
        ),
    ).entity


def test_fresh_evaluation_interfaces_are_public() -> None:
    from photomatagent.scientific import evolution

    assert evolution.FreshEvaluationClaim is not None
    assert evolution.RuntimeFactory is not None
    assert evolution.run_fresh_evaluation is run_fresh_evaluation


def _fresh_ready_service(tmp_path: Path, *, evolution_id: str = "evo_fresh") -> tuple[EvolutionService, StrategyVersion]:
    service = EvolutionService(EvolutionStore(Workspace(tmp_path)))
    _complete_initial(service, evolution_id)
    cutoff = datetime(2026, 9, 1, tzinfo=UTC)
    parameters = {"selector": "fixed-v1"}
    strategy_payload = {
        "evolution_id": evolution_id,
        "revision_id": None,
        "arm": "STATIC",
        "reason": "frozen general baseline",
        "parameters": parameters,
    }
    strategy = StrategyVersion(
        strategy_id="strategy_baseline",
        evolution_id=evolution_id,
        arm="STATIC",
        reason="frozen general baseline",
        parameters=parameters,
        strategy_sha256=hashlib.sha256(
            json.dumps(
                strategy_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        cutoff_at=cutoff,
        created_at=cutoff,
    )
    service.store.write_strategy(strategy)
    task = service.get(evolution_id)
    service.store.save_task(
        task.model_copy(
            update={
                "status": "REVISION_READY",
                "strategy_ids": [strategy.strategy_id],
            }
        ),
        expected_revision=task.revision,
    )
    return service, strategy


def test_fresh_claim_binds_preexisting_strategy_without_feedback_or_revision(
    tmp_path: Path,
) -> None:
    service, strategy = _fresh_ready_service(tmp_path)

    claim = service.claim_fresh_evaluation(
        "evo_fresh",
        strategy_id=strategy.strategy_id,
        owner_token="owner_fresh",
        provider="fake",
        model="fake",
    )

    assert claim.episode.execution_mode == "FRESH_EVALUATION"
    assert claim.episode.parent_version == "v001"
    assert claim.episode.applied_feedback_id is None
    assert claim.episode.revision_plan_id is None
    assert claim.episode.strategy_id == strategy.strategy_id
    assert claim.episode.strategy_sha256 == strategy.strategy_sha256
    assert claim.episode.strategy_cutoff_at == strategy.cutoff_at
    assert claim.episode.owner_token == "owner_fresh"
    assert claim.strategy == strategy


def test_fresh_episode_model_requires_complete_isolated_provenance() -> None:
    with pytest.raises(ValidationError, match="frozen strategy"):
        EpisodeRecord(
            evolution_id="evo_invalid_fresh",
            episode_id="ep_invalid_fresh",
            version="v002",
            execution_mode="FRESH_EVALUATION",
            owner_token="owner_invalid_fresh",
            strategy_id="strategy_invalid_fresh",
            task_snapshot={"goal": "test"},
            target_snapshot=TargetSpec(goal="test"),
        )

    with pytest.raises(ValidationError, match="feedback or revision"):
        EpisodeRecord(
            evolution_id="evo_invalid_fresh",
            episode_id="ep_invalid_fresh",
            version="v002",
            execution_mode="FRESH_EVALUATION",
            owner_token="owner_invalid_fresh",
            strategy_id="strategy_invalid_fresh",
            strategy_sha256="a" * 64,
            strategy_cutoff_at=datetime(2026, 9, 1, tzinfo=UTC),
            applied_feedback_id="fb_invalid",
            task_snapshot={"goal": "test"},
            target_snapshot=TargetSpec(goal="test"),
        )


def test_fresh_claim_is_atomic_and_only_one_owner_wins(tmp_path: Path) -> None:
    service, strategy = _fresh_ready_service(tmp_path)
    barrier = Barrier(2)

    def claim(owner: str):  # type: ignore[no-untyped-def]
        barrier.wait()
        try:
            return service.claim_fresh_evaluation(
                "evo_fresh",
                strategy_id=strategy.strategy_id,
                owner_token=owner,
            )
        except Exception as exc:  # noqa: BLE001 - asserting race result
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(claim, ["owner_a", "owner_b"]))

    assert sum(not isinstance(item, Exception) for item in results) == 1
    assert sum(isinstance(item, EvolutionOperationConflict) for item in results) == 1
    task = service.get("evo_fresh")
    episode = service.store.load_episode("evo_fresh", "v002")
    assert task.status == "RUNNING"
    assert episode.owner_token in {"owner_a", "owner_b"}


def test_fresh_claim_rejects_unlinked_or_unfrozen_strategy(tmp_path: Path) -> None:
    service, strategy = _fresh_ready_service(tmp_path)
    unlinked = strategy.model_copy(update={"strategy_id": "strategy_unlinked"})
    service.store.write_strategy(unlinked)

    with pytest.raises(InvalidEvolutionTransition, match="linked"):
        service.claim_fresh_evaluation(
            "evo_fresh",
            strategy_id=unlinked.strategy_id,
            owner_token="owner_unlinked",
        )

    path = service.store.workspace.resolve(
        ".photomatagent/evolutions/evo_fresh/strategies/strategy_baseline.json",
        must_exist=True,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["strategy_sha256"] = None
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(EvolutionOperationConflict, match="frozen"):
        service.claim_fresh_evaluation(
            "evo_fresh",
            strategy_id=strategy.strategy_id,
            owner_token="owner_unfrozen",
        )


def test_fresh_claim_rejects_strategy_content_that_no_longer_matches_hash(
    tmp_path: Path,
) -> None:
    service, strategy = _fresh_ready_service(tmp_path)
    path = service.store.workspace.resolve(
        ".photomatagent/evolutions/evo_fresh/strategies/strategy_baseline.json",
        must_exist=True,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["reason"] = "tampered strategy content"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(EvolutionOperationConflict, match="hash"):
        service.claim_fresh_evaluation(
            "evo_fresh",
            strategy_id=strategy.strategy_id,
            owner_token="owner_tampered",
        )


@pytest.mark.asyncio
async def test_run_fresh_evaluation_uses_blank_runtime_and_no_learning_history(
    tmp_path: Path,
) -> None:
    service, strategy = _fresh_ready_service(tmp_path)
    task = service.get("evo_fresh")
    observed: dict[str, object] = {}

    def runtime_factory(**kwargs):  # type: ignore[no-untyped-def]
        observed.update(kwargs)
        state = kwargs["scientific_state"]
        observed["initial_state"] = state.model_copy(deep=True)
        runtime = AgentRuntime(
            model=FakeModelProvider([FakeResponse(text="fresh result")]),
            tools=ToolRegistry(),
            workspace=Workspace(tmp_path),
            scientific_state=state,
            permission_policy=DenyAllPolicy(),
            budget=BudgetState(max_iterations=10),
            session_id="session_fresh_evaluation",
        )
        observed["runtime"] = runtime
        return runtime

    result = await run_fresh_evaluation(
        service=service,
        task=task,
        strategy_id=strategy.strategy_id,
        executor=ScientificEpisodeExecutor(service.store),
        runtime_factory=runtime_factory,
        config=ScientificLoopConfig(max_rounds=1),
        owner_token="owner_run_fresh",
    )

    assert result.episode.execution_mode == "FRESH_EVALUATION"
    assert observed["fresh_approval"] is True
    assert str(observed["application_approval_root"]).endswith(
        f"v002_{result.episode.episode_id}"
    )
    assert observed["initial_state"] == ScientificState()
    assert result.episode.revision_plan_id is None
    runtime = observed["runtime"]
    assert isinstance(runtime, AgentRuntime)
    request_text = "\n".join(
        message.model_dump_json() for message in runtime.conversation_state.messages
    )
    assert "Frozen evaluation strategy: STATIC" in request_text
    assert "task-specific history" not in request_text
    assert service.get("evo_fresh").comparison_ids == []
    assert service.get("evo_fresh").experience_ids == []


def test_export_defaults_to_metadata_and_hashes_then_content_is_redacted(
    tmp_path: Path,
) -> None:
    service, strategy = _fresh_ready_service(tmp_path)
    episode = service.store.load_episode("evo_fresh", "v001")
    task = service.get("evo_fresh")
    service.store.save_task(
        task.model_copy(update={"status": "AWAITING_EXPERT_FEEDBACK"}),
        expected_revision=task.revision,
    )
    feedback = service.attach_feedback(
        "evo_fresh",
        episode.version,
        feedback_id="fb_export",
        draft=ExpertFeedbackDraft(
            scores=RubricScores(
                scientific_correctness=3,
                evidence_sufficiency=3,
                novelty=3,
                actionability=3,
                overall=3,
            ),
            comments="password=expert-comment-secret",
        ),
        result_sha256=episode.artifact.sha256,  # type: ignore[union-attr]
        raw_input="Authorization: Bearer expert-raw-secret",
    ).entity
    task = service.get("evo_fresh")
    service.store.save_task(
        task.model_copy(update={"strategy_ids": [strategy.strategy_id]}),
        expected_revision=task.revision,
    )

    metadata_path = service.export_evolution(
        "evo_fresh",
        output=tmp_path / "user_output/export-metadata.json",
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["task"]["evolution_id"] == "evo_fresh"
    assert metadata["episodes"][0]["artifact"]["sha256"] == episode.artifact.sha256  # type: ignore[union-attr]
    assert metadata["feedback"][0]["feedback_id"] == feedback.feedback_id
    assert "raw_input" not in metadata["feedback"][0]
    assert "comments" not in metadata["feedback"][0]
    assert "artifact_content" not in metadata["episodes"][0]
    assert metadata["record_hashes"]["strategies"][strategy.strategy_id]
    assert "expert-raw-secret" not in metadata_path.read_text(encoding="utf-8")
    with pytest.raises(EvolutionAlreadyExistsError):
        service.export_evolution("evo_fresh", output=metadata_path)

    content_path = service.export_evolution(
        "evo_fresh",
        output=tmp_path / "user_output/export-content.json",
        include_content=True,
    )
    serialized = content_path.read_text(encoding="utf-8")
    content = json.loads(serialized)
    assert content["episodes"][0]["artifact_content"]["encoding"] == "utf-8"
    assert content["episodes"][0]["artifact_content"]["body"] == "initial result\n"
    assert content["feedback"][0]["raw_input"] == "Authorization: Bearer [REDACTED]"
    assert "expert-raw-secret" not in serialized
    assert "expert-comment-secret" not in serialized


def test_export_rejects_output_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service, _strategy = _fresh_ready_service(workspace)

    with pytest.raises(Exception, match="outside workspace"):
        service.export_evolution(
            "evo_fresh",
            output=tmp_path / "outside.json",
        )


def test_accept_stop_and_reopen_preserve_exact_checkpoints(tmp_path: Path) -> None:
    service = EvolutionService(EvolutionStore(Workspace(tmp_path)))
    completed = _complete_initial(service, "evo_controls")

    accepted = service.accept("evo_controls", completed.version).entity
    assert accepted.status == "ACCEPTED"
    assert accepted.accepted_version == "v001"
    assert service.reopen("evo_controls").entity.status == "AWAITING_EXPERT_FEEDBACK"

    stopped = service.stop("evo_controls").entity
    assert stopped.status == "STOPPED"
    assert stopped.resume_status == "AWAITING_EXPERT_FEEDBACK"
    assert service.reopen("evo_controls").entity.status == "AWAITING_EXPERT_FEEDBACK"


def test_accept_can_select_any_completed_artifact_not_only_latest(tmp_path: Path) -> None:
    service, strategy = _fresh_ready_service(tmp_path, evolution_id="evo_accept_old")
    claim = service.claim_fresh_evaluation(
        "evo_accept_old",
        strategy_id=strategy.strategy_id,
        owner_token="owner_accept_old",
    )
    service.fail_episode(
        "evo_accept_old",
        claim.episode.version,
        "evaluation stopped",
        owner_token=claim.owner_token,
    )
    service.reopen("evo_accept_old")

    accepted = service.accept("evo_accept_old", "v001").entity

    assert accepted.status == "ACCEPTED"
    assert accepted.accepted_version == "v001"
